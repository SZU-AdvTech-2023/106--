from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from Modules.Distribution import Gaussian
from Modules.Interpolation import ExponentiationLayer, GaussianSmoothing2D
from Modules.Loss import LOSSDICT

from .BaseNetwork import GenerativeRegistrationNetwork
from .ConvBlock import ConvBlock, ConvLayer, ConvNullBlock, EncoderConvModule


class KrebsDecoderConvModule(nn.Module):
    def __init__(self, in_dim, num_layers=[0, 0, 0], dims=[32, 32, 32]):
        super(KrebsDecoderConvModule, self).__init__()
        self.conv0 = ConvLayer(in_dim, dims[0])
        blocklist = []
        for i in range(len(num_layers)):
            block = []
            if num_layers[i]:
                cblock = ConvBlock(num_layers[i], dims[i], dims[i] + 1)
                block.append(('convblock%d' % (i + 1), cblock))
            if i != len(num_layers) - 1:
                up = nn.ConvTranspose2d(dims[i] + (0 if num_layers[i] else 1),
                                        dims[i + 1],
                                        3,
                                        stride=2,
                                        padding=1,
                                        output_padding=1)
                block.append(('up%d' % (i + 1), up))
                relu = nn.LeakyReLU(0.2, inplace=True)
                block.append(('urelu%d' % (i + 1), relu))
            if len(block):
                blocklist.append(nn.Sequential(OrderedDict(block)))
            else:
                blocklist.append(ConvNullBlock())
        self.blocklist = nn.ModuleList(blocklist)

    def forward(self, x, srclist):
        x = self.conv0(x)
        for block, src in zip(self.blocklist, srclist):
            x = torch.cat([x, src], 1)
            x = block(x)
        return x


class KrebsBaseEncoder(nn.Module):
    def __init__(self, z_dim, num_layers, dims, last_block_dim=[]):
        super(KrebsBaseEncoder, self).__init__()
        self.feature = EncoderConvModule(2, num_layers, dims)
        self.qz = Gaussian(16 * 16 * 4, z_dim)
        if len(last_block_dim):
            self.last_block = nn.Sequential(
                OrderedDict([('convlayer0',
                              ConvLayer(dims[-1], last_block_dim[0]))]))
            for i in range(len(last_block_dim) - 1):
                self.last_block.add_module(('convlayer%d' % (i + 1),
                                            ConvLayer(dims[i],
                                                      last_block_dim[i + 1])))
        else:
            self.last_block = ConvNullBlock()

    def forward(self, src, tgt):
        x_in = torch.cat((src, tgt), 1)
        x = self.feature(x_in)
        x = self.last_block(x)
        qz_param = self.qz(torch.flatten(x, start_dim=1))
        return qz_param

    def kld(self, *z_distribution_parm, dim=1):
        return self.qz.kld(*z_distribution_parm, dim=dim)


class KrebsBaseDecoder(nn.Module):
    def __init__(self, size_z, num_layers, dims, last_block_dim=[16, 8, 4]):
        super(KrebsBaseDecoder, self).__init__()
        self.fc0 = nn.Linear(size_z, 1024)
        self.downs1 = nn.AvgPool2d((2, 2))
        self.downs0 = nn.AvgPool2d((4, 4))
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.feature = KrebsDecoderConvModule(1, num_layers, dims)
        self.last_block = nn.Sequential(
            OrderedDict([('convlayer0',
                          ConvLayer(dims[-1] + (0 if num_layers[-1] else 1),
                                    last_block_dim[0]))]))
        for i in range(len(last_block_dim) - 1):
            self.last_block.add_module(
                'convlayer%d' % (i + 1),
                ConvLayer(last_block_dim[i], last_block_dim[i + 1]))
        self.conv4 = nn.Conv2d(last_block_dim[-1], 2, 3, padding=1)

    def forward(self, src, z):
        srclist = [self.downs0(src), self.downs1(src), src]
        x0 = self.relu(self.fc0(z)).view(-1, 1, 32, 32)
        # x0 = self.relu(self.fc0(z)).view(-1, 4, 16, 16)

        x = self.feature(x0, srclist)
        x = self.last_block(x)
        flow = self.conv4(x)

        return flow


class KrebsBaseNet(GenerativeRegistrationNetwork):
    def __init__(self,
                 z_dim,
                 encoder_param,
                 decoder_param,
                 i_size=(128, 128),
                 similarity_factor=5000,
                 similarity_loss='LCC',
                 similarity_loss_param={}):
        super(KrebsBaseNet, self).__init__(i_size)

        self.encoder = KrebsBaseEncoder(z_dim, **encoder_param)
        self.decoder = KrebsBaseDecoder(z_dim, **decoder_param)

        self.similarity_loss = LOSSDICT[similarity_loss](
            **similarity_loss_param)
        self.similarity_factor = similarity_factor

        name = str(similarity_loss) + str(self.similarity_factor)
        for k in similarity_loss_param:
            name += str(similarity_loss_param[k])
        self.name = name

    def sample(self, z_param):
        return self.encoder.qz.sample_z(*z_param)

    def forward(self, src, tgt):
        z_parm = self.encoder(src, tgt)
        z = self.sample(z_parm)
        flow = self.decoder(src, z)
        M_s = self.transformer(src, flow)
        return flow, M_s, z_parm

    def test(self, src, tgt):
        z_parm = self.encoder(src, tgt)
        flow = self.decoder(src, z_parm[0])
        M_s = self.transformer(src, flow)
        return flow, M_s, z_parm

    def objective(self, src, tgt):
        _, src_s, z_param, = self(src, tgt)

        similarity_loss = self.similarity_loss(src_s, tgt)
        kld = self.encoder.kld(*z_param)
        loss = similarity_loss + kld / self.similarity_factor

        return {'loss': loss, 'similarity_loss': similarity_loss, 'kld': kld}

    def uncertainty(self, src, tgt, K):
        z_parm = self.encoder(src, tgt)
        flow_list = []
        for k in range(K):
            z = self.encoder.qz.sample_z(*z_parm)
            flow = self.decoder(src, z)
            smooth_flow = self.smooth(flow)
            exped_flow = self.exp(smooth_flow, self.times - 1)
            flow_list.append(exped_flow)
        flow = torch.stack(flow_list, 2)
        flow_mean = torch.mean(flow, dim=2).unsqueeze(2)
        flow_var = torch.sum(torch.pow(flow - flow_mean, 2), dim=2) / (K - 1)
        h = 0.5 * torch.log(2 * np.pi * flow_var)
        h = torch.mean(h, dim=1)
        return h


class KrebsmoothNet(KrebsBaseNet):
    def __init__(self,
                 z_dim,
                 encoder_param,
                 decoder_param,
                 i_size=(128, 128),
                 similarity_factor=5000,
                 similarity_loss='LCC',
                 similarity_loss_param={},
                 smooth_kernel_size=15,
                 smooth_sigma=3):
        super(KrebsmoothNet,
              self).__init__(z_dim, encoder_param, decoder_param, i_size,
                             similarity_factor, similarity_loss,
                             similarity_loss_param)
        self.smooth = GaussianSmoothing2D(kernel_size=smooth_kernel_size,
                                          sigma=smooth_sigma)

    def forward(self, src, tgt):
        z_parm = self.encoder(src, tgt)
        z = self.sample(*z_parm)
        flow = self.decoder(src, z)
        smooth_flow = self.smooth(flow)
        M_s = self.transformer(src, smooth_flow)
        return smooth_flow, M_s, z_parm

    def test(self, src, tgt):
        z_parm = self.encoder(src, tgt)
        flow = self.decoder(src, z_parm[0])
        smooth_flow = self.smooth(flow)
        M_s = self.transformer(src, smooth_flow)
        return smooth_flow, M_s, z_parm


class KrebsDiffNet(KrebsmoothNet):
    def __init__(self,
                 z_dim,
                 encoder_param,
                 decoder_param,
                 i_size=(128, 128),
                 similarity_factor=5000,
                 similarity_loss='LCC',
                 similarity_loss_param={},
                 smooth_kernel_size=15,
                 smooth_sigma=3,
                 factor=4):
        super(KrebsDiffNet,
              self).__init__(z_dim, encoder_param, decoder_param, i_size,
                             similarity_factor, similarity_loss,
                             similarity_loss_param, smooth_kernel_size,
                             smooth_sigma)
        self.exp = ExponentiationLayer(i_size, factor)
        self.times = factor
        self.name = self.name + '-' + str(self.times)

    def forward(self, src, tgt):
        z_parm = self.encoder(src, tgt)
        z = self.encoder.qz.sample_z(*z_parm)
        flow = self.decoder(src, z)
        smooth_flow = self.smooth(flow)
        exped_flow = self.exp(smooth_flow, self.times - 1)
        M_s = self.transformer(src, exped_flow)
        exped_flow_r = self.exp(-smooth_flow, self.times - 1)
        F_s = self.transformer(tgt, exped_flow_r)
        return exped_flow, M_s, F_s, z_parm

    def test(self, src, tgt):
        z_parm = self.encoder(src, tgt)
        flow = self.decoder(src, z_parm[0])
        smooth_flow = self.smooth(flow)
        exped_flow = self.exp(smooth_flow, self.times)
        M_s = self.transformer(src, exped_flow)
        return exped_flow, M_s, z_parm

    def objective(self, src, tgt):
        _, M_s, F_s, z_parm = self(src, tgt)

        similarity_loss = self.similarity_loss(M_s, F_s)
        kld = self.encoder.kld(*z_parm)

        return {
            'loss': similarity_loss + kld / self.similarity_factor,
            'similarity_loss': similarity_loss,
            'kld': kld
        }

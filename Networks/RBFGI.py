from functools import reduce

import torch
import torch.nn as nn
from Modules.Distribution import Gaussian2D
from Modules.Interpolation import RadialBasisLayer
from Modules.Loss import LOSSDICT, RBFBendingEnergyLoss

from .BaseNetwork import GenerativeRegistrationNetwork
from .ConvBlock import ConvBlock, ConvLayer


class RBFGIInterEncoder(nn.Module):
    def __init__(self,
                 dims=[16, 32, 32, 32, 32],
                 num_layers=[1, 1, 1, 1, 1],
                 local_dims=[16, 32, 32, 32, 32],
                 local_num_layers=[1, 1, 1, 1, 1]):
        super(RBFGIInterEncoder, self).__init__()
        self.cb0 = ConvBlock(num_layers[0], dims[0], 2)  # 128
        self.do0 = ConvLayer(dims[0], dims[1], 3, 2, 1)  # 64
        self.cb1 = ConvBlock(num_layers[1], dims[1], dims[1])
        self.do1 = ConvLayer(dims[1], dims[2], 3, 2, 1)  # 32
        self.cb2 = ConvBlock(num_layers[2], dims[2], dims[2])
        self.do2 = ConvLayer(dims[2], dims[3], 3, 2, 1)  # 16
        self.cb3 = ConvBlock(num_layers[3], dims[3], dims[3])
        self.do3 = ConvLayer(dims[3], dims[4], 3, 2, 1)  # 8
        self.cb4 = ConvBlock(num_layers[4], dims[4], dims[4])
        # sparse alpha
        self.reduce1 = ConvLayer(dims[4], 8, 3, 1, 1)  # 8
        self.qsalpha = Gaussian2D(8, 2, 3, 1, 1)
        # dense branch
        self.dconv0 = ConvLayer(2, local_dims[0], 3)  #64
        self.dcb0 = ConvBlock(local_num_layers[0], local_dims[0],
                              local_dims[0] + dims[0])
        self.ddo0 = ConvLayer(local_dims[0], local_dims[1], 3, 2, 1)  # 32
        self.dcb1_0 = ConvLayer(local_dims[1] + dims[1],
                                local_dims[1],
                                padding=0)  # 30
        self.dcb1_1 = ConvLayer(local_dims[1], local_dims[1], padding=0)  # 28
        self.ddo1 = ConvLayer(local_dims[1], local_dims[2], 3, 2, 1)  # 14
        self.dcb2 = ConvBlock(local_num_layers[2], local_dims[2],
                              local_dims[2] + dims[2])
        self.ddo2 = ConvLayer(local_dims[2], local_dims[3], 3, padding=0)  # 12
        self.dcb3 = ConvBlock(local_num_layers[3], local_dims[3],
                              local_dims[3] + dims[3])
        self.ddo3 = ConvLayer(local_dims[3], local_dims[4], 3, padding=0)  # 10
        self.dcb4 = ConvBlock(local_num_layers[4], local_dims[4],
                              local_dims[4] + dims[4])
        # dense alpha
        self.reduce2 = ConvLayer(local_dims[4], 8, 3, 1, 1)  # 10
        self.qdalpha = Gaussian2D(8, 2, 3, 1, 1)

    def forward(self, src, tgt):
        x_in = torch.cat((src, tgt), 1)
        x0 = self.cb0(x_in)
        x1 = self.cb1(self.do0(x0))
        x2 = self.cb2(self.do1(x1))
        x3 = self.cb3(self.do2(x2))
        x4 = self.cb4(self.do3(x3))
        # sparse alpha
        re = self.reduce1(x4)
        pqsalpha = self.qsalpha(re)
        # dense branch
        # 64
        dx0 = self.dconv0(x_in[:, :, 32:96, 32:96])
        dx0 = torch.cat([dx0, x0[:, :, 32:96, 32:96]], 1)
        dx0 = self.dcb0(dx0)
        # 32
        dx1 = self.ddo0(dx0)
        dx1 = torch.cat([dx1, x1[:, :, 16:48, 16:48]], 1)
        dx1 = self.dcb1_0(dx1)  # 30
        dx1 = self.dcb1_1(dx1)  # 28
        # 14
        dx2_0 = self.ddo1(dx1)
        dx2_1 = torch.nn.functional.interpolate(x2[:, :, 8:24, 8:24], [14, 14],
                                                mode='bilinear')
        dx2 = self.dcb2(torch.cat([dx2_0, dx2_1], 1))
        # 12
        dx3_0 = self.ddo2(dx2)
        dx3_1 = torch.nn.functional.interpolate(x3[:, :, 4:12, 4:12], [12, 12],
                                                mode='bilinear')
        dx3 = self.dcb3(torch.cat([dx3_0, dx3_1], 1))
        # 10
        dx4_0 = self.ddo3(dx3)
        dx4_1 = torch.nn.functional.interpolate(x4[:, :, 2:6, 2:6], [10, 10],
                                                mode='bilinear')
        dx4 = self.dcb4(torch.cat([dx4_0, dx4_1], 1))
        # dense alpha
        dre = self.reduce2(dx4)
        pqdalpha = self.qdalpha(dre)
        return pqsalpha, pqdalpha


class RBFGIGenerativeNetwork(GenerativeRegistrationNetwork):
    def __init__(
        self,
        encoder_param,
        c=2,
        i_size=[128, 128],
        similarity_factor=60000,
        similarity_loss='LCC',
        similarity_loss_param={},
    ):
        super(RBFGIGenerativeNetwork, self).__init__(i_size)
        # used to ouput flow
        vectors = [torch.arange(0, s) for s in i_size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)[[1, 0]]
        grid = grid.unsqueeze(0).float()
        self.register_buffer('grid', grid)
        # control point postion
        cp_loc_vectors = [torch.arange(0, s) for s in (8, 8)]
        cp_loc = torch.meshgrid(cp_loc_vectors)
        cp_loc = torch.stack(cp_loc, 2)[:, :, [1, 0]]
        cp_loc = torch.flatten(cp_loc, start_dim=0, end_dim=1).float()
        # local control point postion
        lcp_loc_vectors = [
            torch.linspace(s, e, 10) for s, e in ((1.5, 5.5), (1.5, 5.5))
        ]
        lcp_loc = torch.meshgrid(lcp_loc_vectors)
        lcp_loc = torch.stack(lcp_loc, 2)[:, :, [1, 0]]
        lcp_loc = torch.flatten(lcp_loc, start_dim=0, end_dim=1).float()
        # combine both
        cp_loc = torch.cat((cp_loc, lcp_loc), 0)
        self.cp_loc = cp_loc
        self.encoder = RBFGIInterEncoder(**encoder_param)
        self.decoder = RadialBasisLayer(cp_loc, i_size, c)
        # loss
        self.similarity_factor = similarity_factor
        self.similarity_loss = LOSSDICT[similarity_loss](
            **similarity_loss_param)
        self.bending_energy = RBFBendingEnergyLoss(cp_loc, c)

        # generate a name
        name = str(similarity_loss) + '--'
        for k in similarity_loss_param:
            name += '-' + str(similarity_loss_param[k])
        name += '--'+str(similarity_factor)
        self.name = name

    def forward(self, src, tgt):
        pqsalpha, pqdalpha = self.encoder(src, tgt)
        # uniform alpha
        salpha = self.encoder.qsalpha.sample_z(*pqsalpha)
        salpha = torch.flatten(salpha, start_dim=2)
        salpha = salpha.permute(0, 2, 1)
        # local alpha
        dalpha = self.encoder.qdalpha.sample_z(*pqdalpha)
        dalpha = torch.flatten(dalpha, start_dim=2)
        dalpha = dalpha.permute(0, 2, 1)
        # combine
        alpha = torch.cat((salpha, dalpha), 1)
        phi = self.decoder(alpha)
        w_src = self.transformer(src, phi)
        return phi, w_src, (pqsalpha, pqdalpha)

    def test(self, src, tgt):
        pqsalpha, pqdalpha = self.encoder(src, tgt)
        # uniform alpha
        salpha = pqsalpha[0]
        salpha = torch.flatten(salpha, start_dim=2)
        salpha = salpha.permute(0, 2, 1)
        # local alpha
        dalpha = pqdalpha[0]
        dalpha = torch.flatten(dalpha, start_dim=2)
        dalpha = dalpha.permute(0, 2, 1)
        # combine
        alpha = torch.cat((salpha, dalpha), 1)
        phi = self.decoder(alpha)
        w_src = self.transformer(src, phi)
        return phi, w_src, (pqsalpha, pqdalpha)

    def objective(self, src, tgt):
        flow, src_s, (pqsalpha, pqdalpha) = self(src, tgt)
        # sigma terms
        sigmas = torch.flatten(pqsalpha[1], start_dim=2).permute(0, 2, 1)
        sigmad = torch.flatten(pqdalpha[1], start_dim=2).permute(0, 2, 1)
        sigma = torch.cat((sigmas, sigmad), 1)

        mus = torch.flatten(pqsalpha[0], start_dim=2).permute(0, 2, 1)
        mud = torch.flatten(pqdalpha[0], start_dim=2).permute(0, 2, 1)
        mu = torch.cat((mus, mud), 1)

        similarity_loss = self.similarity_loss(src_s, tgt)
        sigma_term = torch.sum(torch.exp(sigma) - sigma, dim=[1, 2])
        smooth_term = self.bending_energy(mu)
        KL_loss = sigma_term * 0.5 + smooth_term

        return {
            'loss': similarity_loss + KL_loss / self.similarity_factor,
            'similarity_loss': similarity_loss,
            'KL_loss': KL_loss
        }

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Modules.Interpolation import SpatialTransformer
from Modules.Loss import LOSSDICT
from numpy.core.fromnumeric import ndim

from .BaseNetwork import BaseRegistraionNetwork


class unet_core(nn.Module):
    """
    [unet_core] is a class representing the U-Net implementation that takes in
    a fixed image and a moving image and outputs a flow-field
    """
    def __init__(self, dim, enc_nf, dec_nf, full_size=True):
        """
        Instiatiate UNet model
            :param dim: dimension of the image passed into the net
            :param enc_nf: the number of features maps in each layer of encoding stage
            :param dec_nf: the number of features maps in each layer of decoding stage
            :param full_size: boolean value representing whether full amount of decoding 
                            layers
        """
        super(unet_core, self).__init__()

        self.full_size = full_size
        self.vm2 = len(dec_nf) == 7

        # Encoder functions
        self.enc = nn.ModuleList()
        for i in range(len(enc_nf)):
            prev_nf = 2 if i == 0 else enc_nf[i - 1]
            self.enc.append(conv_block(dim, prev_nf, enc_nf[i], 2))

        # Decoder functions
        self.dec = nn.ModuleList()
        self.dec.append(conv_block(dim, enc_nf[-1], dec_nf[0]))  # 1
        self.dec.append(conv_block(dim, dec_nf[0] * 2, dec_nf[1]))  # 2
        self.dec.append(conv_block(dim, dec_nf[1] * 2, dec_nf[2]))  # 3
        self.dec.append(conv_block(dim, dec_nf[2] + enc_nf[0], dec_nf[3]))  # 4
        self.dec.append(conv_block(dim, dec_nf[3], dec_nf[4]))  # 5

        if self.full_size:
            self.dec.append(conv_block(dim, dec_nf[4] + 2, dec_nf[5], 1))

        if self.vm2:
            self.vm2_conv = conv_block(dim, dec_nf[5], dec_nf[6])

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        """
        Pass input x through the UNet forward once
            :param x: concatenated fixed and moving image
        """
        # Get encoder activations
        x_enc = [x]
        for l in self.enc:
            x_enc.append(l(x_enc[-1]))

        # Three conv + upsample + concatenate series
        y = x_enc[-1]
        for i in range(3):
            y = self.dec[i](y)
            y = self.upsample(y)
            y = torch.cat([y, x_enc[-(i + 2)]], dim=1)

        # Two convs at full_size/2 res
        y = self.dec[3](y)
        y = self.dec[4](y)

        # Upsample to full res, concatenate and conv
        if self.full_size:
            y = self.upsample(y)
            y = torch.cat([y, x_enc[0]], dim=1)
            y = self.dec[5](y)

        # Extra conv for vm2
        if self.vm2:
            y = self.vm2_conv(y)

        return y


class conv_block(nn.Module):
    """
    [conv_block] represents a single convolution block in the Unet which
    is a convolution based on the size of the input channel and output
    channels and then preforms a Leaky Relu with parameter 0.2.
    """
    def __init__(self, dim, in_channels, out_channels, stride=1):
        """
        Instiatiate the conv block
            :param dim: number of dimensions of the input
            :param in_channels: number of input channels
            :param out_channels: number of output channels
            :param stride: stride of the convolution
        """
        super(conv_block, self).__init__()

        conv_fn = getattr(nn, "Conv{0}d".format(dim))

        if stride == 1:
            ksize = 3
        elif stride == 2:
            ksize = 4
        else:
            raise Exception('stride must be 1 or 2')

        self.main = conv_fn(in_channels, out_channels, ksize, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        Pass the input through the conv_block
        """
        out = self.main(x)
        out = self.activation(out)
        return out


class DalcaDiffNet(BaseRegistraionNetwork):
    def __init__(
            self,
            vol_size,
            enc_nf,
            dec_nf,
            #  image_sigma,
            prior_lambda,
            similarity_loss='MSE',
            similarity_loss_param={},
            similarity_factor=30,
            int_steps=7,
            vel_resize=1 / 2,
            bidir=False):
        super(DalcaDiffNet, self).__init__(vol_size)
        self.ndims = len(vol_size)
        self.int_steps = int_steps
        self.vel_resize = vel_resize
        # self.image_sigma = image_sigma
        self.prior_lambda = prior_lambda
        self.bidir = bidir
        self.flow_vol_size = [int(i * vel_resize) for i in vol_size]
        self.unet_model = unet_core(self.ndims,
                                    enc_nf,
                                    dec_nf,
                                    full_size=False)
        conv_fn = getattr(nn, 'Conv%dd' % self.ndims)

        self.flow_mean = conv_fn(dec_nf[-2], self.ndims, 3, 1, 1)
        self.flow_log_sigma = conv_fn(dec_nf[-2], self.ndims, 3, 1, 1)
        self.flow_transformer = SpatialTransformer(self.flow_vol_size)
        self.register_buffer('D', self._degree_matrix(self.flow_vol_size))

        self.similarity_loss = LOSSDICT[similarity_loss](
            **similarity_loss_param)

        self.loss = similarity_loss
        self.similarity_factor = similarity_factor

        nn.init.normal_(self.flow_mean.weight, mean=0.0, std=1e-5)
        nn.init.normal_(self.flow_log_sigma.weight, mean=0.0, std=1e-10)
        nn.init.constant_(self.flow_log_sigma.bias, -10)

        name = str(similarity_loss)
        name += '-' + str(prior_lambda) + '-' + str(similarity_factor)
        for k in similarity_loss_param:
            name += str(similarity_loss_param[k])
        self.name = name

    def forward(self, src, tgt):
        x_out = self.unet_model(torch.cat([src, tgt], 1))
        flow_mean = self.flow_mean(x_out)
        flow_log_sigma = self.flow_log_sigma(x_out)

        noise = torch.randn(*flow_mean.size(), device=src.device)
        vecf = flow_mean + torch.exp(flow_log_sigma / 2.0) * noise

        # v = vecf / (2**self.int_steps)
        v = vecf #Original Version
        for _ in range(self.int_steps):
            v1 = self.flow_transformer(v, v)
            v = v + v1
        flow = v

        flow = flow * (1 / self.vel_resize)

        flow = F.interpolate(flow,
                             scale_factor=1 / self.vel_resize,
                             mode='bilinear')

        warped_tgt = self.transformer(src, flow)

        if self.bidir:
            # nv = -vecf / (2**self.int_steps)
            nv = -vecf # Original Version
            for _ in range(self.int_steps):
                nv1 = self.flow_transformer(nv, nv)
                nv = nv + nv1
            flow_r = nv

            flow_r = flow_r * (1 / self.vel_resize)

            flow_r = F.interpolate(flow_r,
                                   scale_factor=1 / self.vel_resize,
                                   mode='bilinear')

            warped_src = self.transformer(tgt, flow_r)

            return (flow_mean, flow_log_sigma), (warped_tgt, warped_src), flow

        return (flow_mean, flow_log_sigma), (warped_tgt, _), flow

    def test(self, src, tgt):
        x_out = self.unet_model(torch.cat([src, tgt], 1))
        flow = self.flow_mean(x_out)

        # v = flow / (2**self.int_steps)
        v = flow # Original Verison
        for _ in range(self.int_steps):
            v1 = self.flow_transformer(v, v)
            v = v + v1
        flow = v

        flow = flow * (1 / self.vel_resize)

        flow = F.interpolate(flow,
                             scale_factor=1 / self.vel_resize,
                             mode='bilinear')

        warped_tgt = self.transformer(src, flow)

        return flow, warped_tgt, None

    def _adj_filt(self):
        filt_inner = np.zeros([3] * self.ndims)
        for j in range(self.ndims):
            o = [[1]] * self.ndims
            o[j] = [0, 2]
            filt_inner[np.ix_(*o)] = 1

        filt = np.zeros([self.ndims, self.ndims] + [3] * self.ndims)
        for i in range(self.ndims):
            filt[i, i, ...] = filt_inner
        return filt

    def _degree_matrix(self, vol_shape):
        sz = [self.ndims, *vol_shape]

        conv_fn = getattr(F, 'conv%dd' % self.ndims)

        z = torch.ones(1, *sz)
        filt_tf = torch.as_tensor(self._adj_filt()).float()
        return conv_fn(z, filt_tf, padding=1)

    def prec_loss(self, mean):
        sum = torch.zeros(mean.size()[0], device=mean.device)
        for i in range(self.ndims):
            d = i + 2
            r = [d, *range(d), *range(d + 1, self.ndims + 2)]
            y = mean.permute(*r)
            df = y[1:, ...] - y[:-1, ...]
            sum += torch.mean(df * df, dim=[0, *range(2, self.ndims + 2)])
        return 0.5 * self.prior_lambda * sum

    def kl_loss(self, mean, log_sigma):
        sigma_term = self.prior_lambda * self.D * torch.exp(
            log_sigma) - log_sigma
        sigma_term = torch.mean(sigma_term, dim=[*range(1, self.ndims + 2)])

        prec_term = self.prec_loss(mean)

        return 0.5 * (sigma_term + prec_term)

    def recon_loss(self, y_true, y_pred):
        # if self.loss is 'MSE':
        #     df = y_true - y_pred
        #     return 1. / (self.image_sigma**2) * torch.mean(
        #         df * df, dim=[*range(1, self.ndims + 2)])
        # elif self.loss is 'DLCCP':
        #     return (self.similarity_loss(y_true, y_pred)) * self.loss_lambda
        return self.similarity_loss(y_true, y_pred) * self.similarity_factor

    def objective(self, src, tgt):
        (mean, log_sigma), (warped_src, warped_tgt), flow = self(src, tgt)
        recon_loss = self.recon_loss(warped_src, tgt)

        kl_loss = self.kl_loss(mean, log_sigma)
        if self.bidir:
            recon_loss1 = self.recon_loss(warped_tgt, src)
            loss = 0.5 * recon_loss + 0.5 * recon_loss1 + kl_loss

            return {
                'loss': loss,
                'recon_loss': recon_loss,
                'recon_loss1': recon_loss1,
                'kl_loss': kl_loss
            }

        loss = recon_loss + kl_loss

        return {'loss': loss, 'recon_loss': recon_loss, 'kl_loss': kl_loss}

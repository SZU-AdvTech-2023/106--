import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ConvBlock import ConvBlock, ConvLayer, ConvNullBlock
from Modules.Interpolation import SpatialTransformer, RadialBasisArbitraryLayer
from .BaseNetwork import GenerativeRegistrationNetwork
from Modules.Loss import LOSSDICT, JacobianDeterminantLoss, MaxMinPointDist


class OffsetModule(nn.Module):
    def __init__(self, feature_dim, layer_num, input_dim, output_dim,
                 feature_size):
        super(OffsetModule, self).__init__()
        if layer_num:
            self.cp_feature = ConvBlock(layer_num, feature_dim, input_dim)
            self.os_feature = ConvBlock(layer_num, feature_dim, input_dim)
        else:
            self.cp_feature = ConvNullBlock()
            self.os_feature = ConvNullBlock()

        self.cp_offset = nn.Conv2d(feature_dim, 2, 3, stride=1, padding=1)
        self.transformer = SpatialTransformer(feature_size, need_grid='True')
        self.cp_feature_ds = ConvLayer(feature_dim * 2, output_dim, stride=2)
        self.os_feature_ds = ConvLayer(feature_dim, output_dim, stride=2)

    def forward(self, pre_cp_feature, pre_os_feature, align_corners):
        cp_feature = self.cp_feature(pre_cp_feature)
        cp_offset = self.cp_offset(cp_feature)

        os_feature = self.os_feature(pre_os_feature)
        warped_os_feature = self.transformer(os_feature,
                                             cp_offset,
                                             align_corners=align_corners)

        new_cp_feature = self.cp_feature_ds(
            torch.cat([cp_feature, warped_os_feature], 1))
        new_os_feature = self.os_feature_ds(os_feature)

        return new_cp_feature, new_os_feature


class OutputModuleP(nn.Module):
    def __init__(self, feature_dim, layer_num, input_dim, feature_size):
        super(OutputModuleP, self).__init__()
        self.cp_feature = ConvBlock(layer_num, feature_dim, input_dim)
        self.cp_offset = nn.Conv2d(feature_dim, 2, 3, stride=1, padding=1)

        self.os_feature = ConvBlock(layer_num, feature_dim, input_dim)
        self.transformer = SpatialTransformer(feature_size, need_grid='True')

        self.warp_os_feature = ConvBlock(layer_num, feature_dim, input_dim)
        self.offset = nn.Conv2d(feature_dim, 2, 3, stride=1, padding=1)
        self.sigma = nn.Conv2d(feature_dim, 2, 3, stride=1, padding=1)
        torch.nn.init.normal_(self.sigma.weight, mean=0.0, std=1e-10)
        torch.nn.init.constant_(self.sigma.bias, -10)

    def forward(self, pre_cp_feature, pre_os_feature, align_corners):
        cp_feature = self.cp_feature(pre_cp_feature)
        cp_offset = self.cp_offset(cp_feature)

        os_feature = self.os_feature(pre_os_feature)
        warped_os_feature = self.transformer(os_feature,
                                             cp_offset,
                                             align_corners=align_corners)

        warped_os_feature = self.warp_os_feature(warped_os_feature)
        offset = self.offset(warped_os_feature)
        sigma = self.sigma(warped_os_feature)
        return cp_offset, (offset, sigma)


class RBFDCGenerativeEncoder(nn.Module):
    def __init__(self, feature_dims, layer_nums, i_size=[128, 128]):
        super(RBFDCGenerativeEncoder, self).__init__()
        offsetList = []
        for i in range(len(feature_dims) - 1):
            scale = 2**i
            offsetList.append(
                OffsetModule(feature_dims[i], layer_nums[i],
                             feature_dims[i] if i else 2, feature_dims[i + 1],
                             [i_size[0] // scale, i_size[1] // scale]))
        self.offsetList = nn.ModuleList(offsetList)
        final_i = len(feature_dims) - 1
        final_scale = 2**(final_i)
        self.output = OutputModuleP(
            feature_dims[final_i], layer_nums[final_i], feature_dims[final_i],
            (i_size[0] // final_scale, i_size[1] // final_scale))

    def forward(self, src, tgt, align_corners):
        x = (torch.cat([src, tgt], 1), torch.cat([src, tgt], 1))
        for l in self.offsetList:
            x = l(*x, align_corners)
        cp_offset, pqoffset = self.output(*x, align_corners)
        return cp_offset, pqoffset


class RBFDCGenerativeNetwork(GenerativeRegistrationNetwork):
    def __init__(self,
                 encoder_param,
                 i_size,
                 c_factor,
                 cpoint_num,
                 align_corners=False,
                 similarity_loss='LCC',
                 similarity_loss_param={},
                 factor_list=[130000, 150, 1]):
        super(RBFDCGenerativeNetwork, self).__init__(i_size)
        self.encoder = RBFDCGenerativeEncoder(**encoder_param)
        self.decoder = RadialBasisArbitraryLayer(i_size, c_factor, cpoint_num)
        self.scale = int(i_size[0] // np.sqrt(cpoint_num))
        self.transformer = SpatialTransformer(i_size, need_grid=True)
        # initialize the positions of control points
        cp_loc_vectors = [
            torch.linspace(self.scale // 2, i - self.scale // 2,
                           i // self.scale) for i in i_size
        ]
        cp_loc = torch.meshgrid(cp_loc_vectors)
        cp_loc = torch.stack(cp_loc, 2)[:, :, [1, 0]]
        cp_loc = torch.flatten(cp_loc, start_dim=0, end_dim=1).float()
        self.register_buffer('cp_grid', cp_loc)
        self.i_size = i_size

        self.c_factor = c_factor
        self.cpoint_num = cpoint_num
        self.align_corners = align_corners

        self.similarity_loss = LOSSDICT[similarity_loss](
            **similarity_loss_param)
        self.jacobian_loss = JacobianDeterminantLoss()
        self.cpoint_maxmin = MaxMinPointDist(cpoint_num)
        self.factor_list = factor_list

        # generate a name
        name = str(similarity_loss) + '--'
        for k in similarity_loss_param:
            name += '-' + str(similarity_loss_param[k])
        name += '--'
        for i in self.factor_list:
            name += str(i)
        self.name = name

    def sample(self, pqoffset):
        eps = torch.randn(pqoffset[0].size(), device=pqoffset[0].device)
        std = torch.exp(0.5 * pqoffset[1])
        return pqoffset[0] + std * eps

    def forward(self, src, tgt):
        scp_offset, pqoffset = self.encoder(src,
                                            tgt,
                                            align_corners=self.align_corners)
        # rescale to final resolution and reshape
        scp_offset = scp_offset * self.scale
        cp_offset = torch.flatten(scp_offset.permute(0, 2, 3, 1),
                                  start_dim=1,
                                  end_dim=2)
        cp_loc = self.cp_grid + cp_offset
        # reshape offset
        alpha = self.sample(pqoffset)

        alpha = torch.flatten(alpha.permute(0, 2, 3, 1),
                              start_dim=1,
                              end_dim=2)
        flow = self.decoder(cp_loc, alpha)
        warped_src = self.transformer(src, flow)

        return flow, warped_src, cp_loc, pqoffset

    def test(self, src, tgt):
        cp_offset, pqoffset = self.encoder(src, tgt, self.align_corners)
        # rescale to final resolution and reshape
        cp_offset = cp_offset * self.scale
        cp_offset = torch.flatten(cp_offset.permute(0, 2, 3, 1),
                                  start_dim=1,
                                  end_dim=2)
        cp_loc = self.cp_grid + cp_offset
        # reshape offset
        alpha = pqoffset[0]
        alpha = torch.flatten(alpha.permute(0, 2, 3, 1),
                              start_dim=1,
                              end_dim=2)
        flow = self.decoder(cp_loc, alpha)
        warped_src = self.transformer(src, flow)

        return flow, warped_src, cp_loc, pqoffset

    def smooth_loss(self, alpha, cp_loc):
        c = self.cpoint_maxmin(cp_loc) * self.c_factor  # b
        c = c.unsqueeze(1).unsqueeze(1)
        # b c c 2
        cp_loc_ta = cp_loc.unsqueeze(1).repeat(1, self.cpoint_num, 1, 1)
        cp_loc_tb = cp_loc.unsqueeze(2).repeat(1, 1, self.cpoint_num, 1)
        dist = torch.norm(cp_loc_ta - cp_loc_tb, dim=3) / c
        # add mask for r < 1
        mask = dist < 1
        # weight if r<1 weight=(1-r)^4*(4r+1)
        #        else   weight=0
        # Todo: reduce weight size
        weight = torch.pow(1 - dist, 4) * (4 * dist + 1)
        weight = weight * mask.float()  # b c c
        det = torch.det(weight) + 1e-5
        logdet = torch.log(det)
        weight = weight.unsqueeze(3).repeat(1, 1, 1, 2)
        # tile alpha
        alpha_t = alpha.unsqueeze(1).repeat(1, self.cpoint_num, 1, 1)
        y = torch.sum(alpha_t * weight, dim=2)
        K = torch.sum(y * alpha, dim=[1, 2])
        return K, logdet

    def setHyperparam(self, similarity_loss_param, factor_list):
        '''
        for hyperparam optimization
        '''
        self.similarity_loss.set(**similarity_loss_param)
        self.factor_list = factor_list


    def objective(self, src, tgt):
        flow, warped_src, cp_loc, (mu, log_var) = self(src, tgt)
        # sigma terms
        sigma_term = torch.sum(torch.exp(log_var), dim=[1, 2, 3]) - torch.sum(
            log_var, dim=[1, 2, 3])
        # smooth terms
        mu = torch.flatten(mu.permute(0, 2, 3, 1), start_dim=1, end_dim=2)
        smooth_term, logdet = self.smooth_loss(mu, cp_loc)
        # KL
        KL_loss = (sigma_term + smooth_term - logdet) * 0.5

        # similarity
        similarity_loss = self.similarity_loss(warped_src, tgt)

        # L1
        L1_loss = torch.sum(torch.abs(mu), dim=[1, 2])

        # jacobian
        jacobian_loss = self.jacobian_loss(flow)

        loss = self.factor_list[
            0] * similarity_loss + KL_loss + self.factor_list[
                1] * jacobian_loss + self.factor_list[2] * L1_loss

        return {
            'similarity_loss': similarity_loss,
            'KL_loss': KL_loss,
            'jacobian_loss': jacobian_loss,
            'L1_loss': L1_loss,
            'loss': loss
        }

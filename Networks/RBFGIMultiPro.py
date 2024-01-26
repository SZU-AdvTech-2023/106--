import math
from functools import reduce

import torch
import torch.nn as nn
from Modules.Distribution import Gaussian2D
from Modules.Interpolation.SpatialTransformer import SpatialTransformer
from Modules.Loss import (LOSSDICT, JacobianDeterminantLoss,
                          RBFBendingEnergyLossA)
from torch import tensor

from .BaseNetwork import GenerativeRegistrationNetwork
from .ConvBlock import ConvBlock, ConvLayer
# from roi_align import RoIAlign
from mmcv.ops.roi_align import RoIAlign

def Sample(mu, log_var):
    eps = torch.randn(mu.size(), device=mu.device)

    std = torch.exp(0.5 * log_var)

    z = mu + std * eps

    return z


class RBFGISharedEncoderA(nn.Module):
    def __init__(self,                 index=2,

                 dims=[16, 32, 32, 32, 32],
                 num_layers=[1, 1, 1, 1, 1],
                 local_dims=[16, 32, 32, 32, 32],
                 local_num_layers=[1, 1, 1, 1, 1]):
        super(RBFGISharedEncoderA, self).__init__()
        self.index = index
        if index >= 1:
            self.cb0 = ConvBlock(num_layers[0], dims[0], 2)  # G 128

            self.dconv0 = ConvLayer(2, local_dims[0], 3)  # L 64
            self.dcb0 = ConvBlock(local_num_layers[0], local_dims[0],
                                  local_dims[0] + dims[0])

        if index >= 2:
            self.do0 = ConvLayer(dims[0], dims[1], 3, 2, 1)  # G 64
            self.cb1 = ConvBlock(num_layers[1], dims[1], dims[1])

            self.ddo0 = ConvLayer(local_dims[0], local_dims[1], 3, 2,
                                  1)  # L 32
            self.dcb1_0 = ConvLayer(local_dims[1] + dims[1],
                                    local_dims[1],
                                    padding=0)  # L 30
            self.dcb1_1 = ConvLayer(local_dims[1], local_dims[1],
                                    padding=0)  # L 28

        if index >= 3:
            self.do1 = ConvLayer(dims[1], dims[2], 3, 2, 1)  # G 32
            self.cb2 = ConvBlock(num_layers[2], dims[2], dims[2])

            self.ddo1 = ConvLayer(local_dims[1], local_dims[2], 3, 2, 1)  #L 14
            self.dcb2 = ConvBlock(local_num_layers[2], local_dims[2],
                                  local_dims[2] + dims[2])

        if index >= 4:
            self.do2 = ConvLayer(dims[2], dims[3], 3, 2, 1)  #G 16
            self.cb3 = ConvBlock(num_layers[3], dims[3], dims[3])

            self.ddo2 = ConvLayer(local_dims[2], local_dims[3], 3,
                                  padding=0)  #L 12
            self.dcb3 = ConvBlock(local_num_layers[3], local_dims[3],
                                  local_dims[3] + dims[3])

        self.crop_and_resize0 = RoIAlign(64, 64)
        self.crop_and_resize1 = RoIAlign(32, 32)
        self.crop_and_resize2 = RoIAlign(14, 14)
        self.crop_and_resize3 = RoIAlign(12, 12)
        self.crop_and_resize4 = RoIAlign(10, 10)

    def forward(self, src, tgt, scale):
        scale = scale.unsqueeze(1).repeat(1, 2).view(4)
        batch = src.size()[0]
        box_index = torch.arange(batch).cuda().int()

        x_global = torch.cat((src, tgt), 1)

        box = torch.Tensor([32, 32, 96, 96]).cuda()
        box = (box - 64) * scale + 64
        boxes = box.unsqueeze(0).repeat(batch, 1)
        x_local = self.crop_and_resize0(x_global, boxes, box_index)
        # print(x_local)

        # x_local = x_global[:, :, 32:96, 32:96]

        if self.index >= 1:
            x_global = self.cb0(x_global)
            #
            dx0 = self.dconv0(x_local)
            dx0 = torch.cat(
                [dx0, self.crop_and_resize0(x_global, boxes, box_index)], 1)
            x_local = self.dcb0(dx0)

        if self.index >= 2:
            box = torch.Tensor([16, 16, 48, 48]).cuda()
            box = (box - 32) * scale + 32
            boxes = box.unsqueeze(0).repeat(batch, 1)

            x_global = self.cb1(self.do0(x_global))
            #
            dx1 = self.ddo0(x_local)
            dx1 = torch.cat(
                [dx1, self.crop_and_resize1(x_global, boxes, box_index)], 1)
            dx1 = self.dcb1_0(dx1)  # 30
            x_local = self.dcb1_1(dx1)  # 28

        if self.index >= 3:
            box = torch.Tensor([8, 8, 24, 24]).cuda()
            box = (box - 16) * scale + 16
            boxes = box.unsqueeze(0).repeat(batch, 1)

            x_global = self.cb2(self.do1(x_global))
            #
            dx2_0 = self.ddo1(x_local)
            # dx2_1 = torch.nn.functional.interpolate(x_global[:, :, 8:24, 8:24],
            #                                         [14, 14],
            #                                         mode='bilinear')
            dx2_1 = self.crop_and_resize2(x_global, boxes, box_index)
            x_local = self.dcb2(torch.cat([dx2_0, dx2_1], 1))

        if self.index >= 4:
            box = torch.Tensor([4, 4, 12, 12]).cuda()
            box = (box - 8) * scale + 8
            boxes = box.unsqueeze(0).repeat(batch, 1)

            x_global = self.cb3(self.do2(x_global))
            #
            dx3_0 = self.ddo2(x_local)
            # dx3_1 = torch.nn.functional.interpolate(x_global[:, :, 4:12, 4:12],
            #                                         [12, 12],
            #                                         mode='bilinear')
            dx3_1 = self.crop_and_resize3(x_global, boxes, box_index)
            x_local = self.dcb3(torch.cat([dx3_0, dx3_1], 1))

        return x_global, x_local


class RBFGIUnSharedEncoderA(nn.Module):
    def __init__(self,
                 index=2,
                 dims=[16, 32, 32, 32, 32],
                 num_layers=[1, 1, 1, 1, 1],
                 local_dims=[16, 32, 32, 32, 32],
                 local_num_layers=[1, 1, 1, 1, 1]):
        super(RBFGIUnSharedEncoderA, self).__init__()
        self.index = index
        if index <= 2:
            self.do0 = ConvLayer(dims[0], dims[1], 3, 2, 1)  # G 64
            self.cb1 = ConvBlock(num_layers[1], dims[1], dims[1])

            self.ddo0 = ConvLayer(local_dims[0], local_dims[1], 3, 2,
                                  1)  # L 32
            self.dcb1_0 = ConvLayer(local_dims[1] + dims[1],
                                    local_dims[1],
                                    padding=0)  # L 30
            self.dcb1_1 = ConvLayer(local_dims[1], local_dims[1],
                                    padding=0)  # L 28

        if index <= 3:
            self.do1 = ConvLayer(dims[1], dims[2], 3, 2, 1)  # G 32
            self.cb2 = ConvBlock(num_layers[2], dims[2], dims[2])

            self.ddo1 = ConvLayer(local_dims[1], local_dims[2], 3, 2, 1)  #L 14
            self.dcb2 = ConvBlock(local_num_layers[2], local_dims[2],
                                  local_dims[2] + dims[2])

        if index <= 4:
            self.do2 = ConvLayer(dims[2], dims[3], 3, 2, 1)  #G 16
            self.cb3 = ConvBlock(num_layers[3], dims[3], dims[3])

            self.ddo2 = ConvLayer(local_dims[2], local_dims[3], 3,
                                  padding=0)  #L 12
            self.dcb3 = ConvBlock(local_num_layers[3], local_dims[3],
                                  local_dims[3] + dims[3])

        self.do3 = ConvLayer(dims[3], dims[4], 3, 2, 1)  # 8
        self.cb4 = ConvBlock(num_layers[4], dims[4], dims[4])

        self.ddo3 = ConvLayer(local_dims[3], local_dims[4], 3, padding=0)  # 10
        self.dcb4 = ConvBlock(local_num_layers[4], local_dims[4],
                              local_dims[4] + dims[4])

        self.crop_and_resize1 = RoIAlign(32, 32)
        self.crop_and_resize2 = RoIAlign(14, 14)
        self.crop_and_resize3 = RoIAlign(12, 12)
        self.crop_and_resize4 = RoIAlign(10, 10)

        # global
        self.reduce1 = ConvLayer(dims[4], 8, 3, 1, 1)  # 8
        self.qsalpha = Gaussian2D(8, 2, 3, 1, 1)
        # local
        self.reduce2 = ConvLayer(local_dims[4], 8, 3, 1, 1)  # 10
        self.qdalpha = Gaussian2D(8, 2, 3, 1, 1)

    def forward(self, x_global, x_local, scale):
        scale = scale.unsqueeze(1).repeat(1, 2).view(4)
        batch = x_global.size()[0]
        box_index = torch.arange(batch).cuda().int()

        if self.index <= 2:
            box = torch.Tensor([16, 16, 48, 48]).cuda()
            box = (box - 32) * scale + 32
            boxes = box.unsqueeze(0).repeat(batch, 1)

            x_global = self.cb1(self.do0(x_global))
            #
            dx1 = self.ddo0(x_local)
            dx1 = torch.cat(
                [dx1, self.crop_and_resize1(x_global, boxes, box_index)], 1)
            dx1 = self.dcb1_0(dx1)  # 30
            x_local = self.dcb1_1(dx1)  # 28

        if self.index <= 3:
            box = torch.Tensor([8, 8, 24, 24]).cuda()
            box = (box - 16) * scale + 16
            boxes = box.unsqueeze(0).repeat(batch, 1)

            x_global = self.cb2(self.do1(x_global))
            #
            dx2_0 = self.ddo1(x_local)
            # dx2_1 = torch.nn.functional.interpolate(x_global[:, :, 8:24, 8:24],
            #                                         [14, 14],
            #                                         mode='bilinear')
            dx2_1 = self.crop_and_resize2(x_global, boxes, box_index)
            x_local = self.dcb2(torch.cat([dx2_0, dx2_1], 1))

        if self.index <= 4:
            box = torch.Tensor([4, 4, 12, 12]).cuda()
            box = (box - 8) * scale + 8
            boxes = box.unsqueeze(0).repeat(batch, 1)

            x_global = self.cb3(self.do2(x_global))
            #
            dx3_0 = self.ddo2(x_local)
            # dx3_1 = torch.nn.functional.interpolate(x_global[:, :, 4:12, 4:12],
            #                                         [12, 12],
            #                                         mode='bilinear')
            dx3_1 = self.crop_and_resize3(x_global, boxes, box_index)
            x_local = self.dcb3(torch.cat([dx3_0, dx3_1], 1))
        x_global = self.cb4(self.do3(x_global))
        box = torch.Tensor([2, 2, 6, 6]).cuda()
        box = (box - 4) * scale + 4
        boxes = box.unsqueeze(0).repeat(batch, 1)

        dx4_0 = self.ddo3(x_local)
        # dx4_1 = torch.nn.functional.interpolate(x_global[:, :, 2:6, 2:6],
        #                                         [10, 10],
        #                                         mode='bilinear')
        dx4_1 = self.crop_and_resize4(x_global, boxes, box_index)
        x_local = self.dcb4(torch.cat([dx4_0, dx4_1], 1))

        # global alpha
        re = self.reduce1(x_global)
        pqsalpha_mu, pqsalpha_var = self.qsalpha(re)
        pqsalpha_mu = torch.flatten(pqsalpha_mu, start_dim=2)
        pqsalpha_var = torch.flatten(pqsalpha_var, start_dim=2)
        # local alpha
        dre = self.reduce2(x_local)
        pqdalpha_mu, pqdalpha_var = self.qdalpha(dre)
        pqdalpha_mu = torch.flatten(pqdalpha_mu, start_dim=2)
        pqdalpha_var = torch.flatten(pqdalpha_var, start_dim=2)

        mu = torch.cat([pqsalpha_mu, pqdalpha_mu], 2).permute(0, 2, 1)

        var = torch.cat([pqsalpha_var, pqdalpha_var], 2).permute(0, 2, 1)

        return mu, var


class RBFGIMutilRadiusEncoderA(nn.Module):
    def __init__(self, shared_param, unshared_param, c_nums=3):
        super(RBFGIMutilRadiusEncoderA, self).__init__()
        self.shared_encoder = RBFGISharedEncoderA(**shared_param)

        self.unshared_encoder_list = nn.ModuleList(
            [RBFGIUnSharedEncoderA(**unshared_param) for _ in range(c_nums)])

    def forward(self, src, tgt, scale):
        shared_feature = self.shared_encoder(src, tgt, scale)
        mu_list, var_list = [], []
        for unshared_encoder in self.unshared_encoder_list:
            res = unshared_encoder(*shared_feature, scale)
            mu_list.append(res[0])
            var_list.append(res[1])

        return mu_list, var_list


class RBFGIMutilAdaptiveDecoder(nn.Module):
    def __init__(self, img_size, c_list, int_steps=None):
        super(RBFGIMutilAdaptiveDecoder, self).__init__()

        # self.scale = nn.parameter.Parameter(data=torch.tensor(
        #     [1, 1], dtype=torch.float),
        #                                     requires_grad=True)

        global_cp_loc_grid = [
            torch.linspace(s + (e - s) / 16, e - (e - s) / 16, 8)
            for s, e in ((0, 8), (0, 8))
        ]
        global_cp_loc_grid = torch.meshgrid(global_cp_loc_grid)
        global_cp_loc = torch.stack(global_cp_loc_grid, 2)[:, :, [1, 0]]
        global_cp_loc = torch.flatten(global_cp_loc, start_dim=0,
                                      end_dim=1).float()
        self.register_buffer('global_cp_loc', global_cp_loc)

        local_cp_loc_grid = [
            torch.linspace(s + (s - e) / 20, e - (s - e) / 20, 10)
            for s, e in ((2, 6), (2, 6))
        ]
        local_cp_loc_grid = torch.meshgrid(local_cp_loc_grid)
        local_cp_loc = torch.stack(local_cp_loc_grid, 2)[:, :, [1, 0]]
        local_cp_loc = torch.flatten(local_cp_loc, start_dim=0,
                                     end_dim=1).float()
        self.register_buffer('local_cp_loc', local_cp_loc)
        self.img_size = img_size
        self.c_list = c_list

        # a location mesh of output
        loc_vectors = [torch.linspace(0.0, 8.0, i_s) for i_s in self.img_size]
        loc = torch.meshgrid(loc_vectors)
        loc = torch.stack(loc, 2)
        loc = loc[:, :, [1, 0]].float().unsqueeze(2)
        # repeating for calculate the distance of contorl cpoints
        loc_tile = loc.repeat(1, 1,
                              local_cp_loc.size()[0] + global_cp_loc.size()[0],
                              1)
        self.register_buffer('loc_tile', loc_tile)
        self.int_steps = int_steps

        self.first_test = True

        if self.int_steps:
            self.flow_transformer = SpatialTransformer(self.img_size)

    def cp_gen(self, scale):
        local_cp_loc = (self.local_cp_loc - 4) * scale + 4
        cpoint_pos = torch.cat([self.global_cp_loc, local_cp_loc], 0)
        return cpoint_pos

    def getWeight(self, c, scale):
        local_cp_loc = (self.local_cp_loc - 4) * scale + 4
        cpoint_pos = torch.cat([self.global_cp_loc, local_cp_loc], 0)

        # a location mesh of control points
        cp_loc = cpoint_pos.unsqueeze(0).unsqueeze(0)
        cp_loc_tile = cp_loc.repeat(*self.img_size, 1, 1)

        # calculate r
        dist = torch.norm(self.loc_tile - cp_loc_tile, dim=3) / c
        # add mask for r < 1
        mask = dist < 1
        # weight if r<1 weight=(1-r)^4*(4r+1)
        #        else   weight=0
        # Todo: reduce weight size
        weight = torch.pow(1 - dist, 4) * (4 * dist + 1)
        weight = weight * mask.float()
        weight = weight.unsqueeze(0).unsqueeze(4)
        # print('calculate weight')

        return weight

    def interpolate(self, alpha, c, scale):
        weight = self.getWeight(c, scale)

        alpha = alpha.unsqueeze(1).unsqueeze(1)
        phi = torch.sum(weight * alpha, 3)
        phi = phi.permute(0, 3, 1, 2)
        return phi

    def interpolateForTest(self, all_alpha, scale):
        if self.first_test:
            self.first_test = False
            weight_list = [
                self.getWeight(c, scale).unsqueeze(1) for c in self.c_list
            ]
            self.weight = torch.cat(weight_list, 1)
            # print('weight done')

        all_alpha = all_alpha.unsqueeze(2).unsqueeze(2)
        phi = torch.sum(self.weight * all_alpha, 4)
        return torch.sum(phi, 1).permute(0, 3, 1, 2)

    def diffeomorphic(self, flow):
        v = flow / (2**self.int_steps)
        for _ in range(self.int_steps):
            v1 = self.flow_transformer(v, v)
            v = v + v1
        return v

    def forward(self, src, mu_list, var_list, scale):
        phi_list = []
        for c, mu, var in zip(self.c_list, mu_list, var_list):
            alpha = Sample(mu, var)
            phi_list.append(self.interpolate(alpha, c, scale))

        phi = reduce(torch.add, phi_list)

        if self.int_steps:
            phi = self.diffeomorphic(phi)

        return phi

    def test(self, src, mu_list, var_list, scale):
        phi_list = []
        for c, mu, var in zip(self.c_list, mu_list, var_list):
            alpha = mu
            phi_list.append(self.interpolate(alpha, c, scale))

        phi = reduce(torch.add, phi_list)

        # all_alpha = torch.cat([mu.unsqueeze(1) for mu in mu_list], 1)
        # phi = self.interpolateForTest(all_alpha)

        if self.int_steps:
            phi = self.diffeomorphic(phi)

        return phi


class RBFGIMutilRadiusAAdaptivePro(GenerativeRegistrationNetwork):
    def __init__(self,
                 encoder_param,
                 c_list=[1.5, 2, 2.5],
                 i_size=[128, 128],
                 factor_list=[60000],
                 similarity_loss='LCC',
                 similarity_loss_param={},
                 int_steps=None):
        super(RBFGIMutilRadiusAAdaptivePro, self).__init__(i_size)
        self.scale = nn.parameter.Parameter(data=torch.tensor(
            [1, 1], dtype=torch.float),
                                            requires_grad=True)
        self.encoder = RBFGIMutilRadiusEncoderA(**encoder_param)
        self.decoder = RBFGIMutilAdaptiveDecoder(i_size, c_list, int_steps)
        self.bending_energy_cal = RBFBendingEnergyLossA()
        self.similarity_loss = LOSSDICT[similarity_loss](
            **similarity_loss_param)
        self.jacobian_loss = JacobianDeterminantLoss()
        self.factor_list = factor_list
        self.c_list = c_list

        # generate a name
        name = str(similarity_loss) + '--'
        for k in similarity_loss_param:
            name += '-' + str(similarity_loss_param[k])
        name += '--'
        for i in self.factor_list:
            name += str(i)
        self.name = name
        if int_steps:
            self.name += '-diff'

    def forward(self, src, tgt):
        alpha_list = self.encoder(src, tgt, self.scale)
        phi = self.decoder(src, *alpha_list, self.scale)
        w_src = self.transformer(src, phi)

        return phi, w_src, alpha_list

    def test(self, src, tgt):
        alpha_list = self.encoder(src, tgt, self.scale)
        phi = self.decoder.test(src, *alpha_list, self.scale)

        w_src = self.transformer(src, phi)
        return phi, w_src, alpha_list

    def objective(self, src, tgt):
        flow, src_s, alpha_list = self(src, tgt)

        sigma = torch.cat(alpha_list[1], 1)
        sigma_term = torch.sum(torch.exp(sigma) - sigma, dim=[1, 2])

        bending_energy_list = [
            self.bending_energy_cal(mu, self.decoder.cp_gen(self.scale), c)
            for mu, c in zip(alpha_list[0], self.c_list)
        ]
        smooth_term = reduce(torch.add, bending_energy_list)

        KL_loss = sigma_term / 2 + smooth_term

        similarity_loss = self.similarity_loss(src_s, tgt)

        jacobian_loss = self.jacobian_loss(flow)

        L1_loss = torch.sum(torch.abs(torch.cat(alpha_list[0], 1)), dim=[1, 2])

        return {
            'similarity_loss':
            similarity_loss,
            'KL_loss':
            KL_loss,
            'jacobian_loss':
            jacobian_loss,
            'L1_loss':
            L1_loss,
            'loss':
            self.factor_list[0] * similarity_loss + KL_loss +
            self.factor_list[1] * jacobian_loss +
            self.factor_list[2] * L1_loss,
        }

    def uncertainty(self, src, tgt, K):
        _, _, alpha_list = self(src, tgt)
        d = []
        for _ in range(K):
            phi_xy = self.decoder(src, *alpha_list,
                                  self.scale).unsqueeze(1)  # B 1 2 H W
            phi_r = torch.norm(phi_xy, dim=2, keepdim=True)  # B 1 1 H W
            phi_theta = torch.abs(
                torch.atan2(phi_xy[:, :, 1, :, :],
                            phi_xy[:, :,
                                   0, :, :])).unsqueeze(2) * 180  # B 1 1 H W
            phi_polar = torch.cat([phi_r, phi_theta], dim=2)  # B 1 2 H W
            phi = torch.cat([phi_xy, phi_polar], dim=2)  # B 1 4 H W
            d.append(phi)

        d = torch.cat(d, dim=1)  # B K 4 H W
        d_expect = torch.mean(d, dim=1)  # B 4 H W
        dd_expect = torch.mean(d * d, dim=1)  # B 4 H W
        var1 = 1 + dd_expect - d_expect * d_expect
        # var = torch.mean(torch.pow(d - d_expect, 2), dim=1)
        # return var
        var2 = torch.mean(torch.pow(d - torch.mean(d, dim=1, keepdim=True), 2),
                          dim=1)
        return torch.cat([var1, var2], dim=1), d_expect
from collections import OrderedDict

import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    """
    a combination of conv and relu
    """
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=1):
        super(ConvLayer, self).__init__()
        self.layer = nn.Sequential(
            OrderedDict([('conv',
                          nn.Conv2d(in_dim,
                                    out_dim,
                                    kernel_size,
                                    stride=stride,
                                    padding=padding)),
                         ('relu', nn.LeakyReLU(0.2, inplace=True))]))

    def forward(self, feature):
        return self.layer(feature)


class DeconvLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 kernel_size=3,
                 stride=2,
                 padding=1,
                 output_padding=1):
        super(DeconvLayer, self).__init__()
        self.layer = nn.Sequential(
            OrderedDict([('deconv',
                          nn.ConvTranspose2d(in_dim,
                                             out_dim,
                                             kernel_size,
                                             stride=stride,
                                             padding=padding,
                                             output_padding=output_padding)),
                         ('relu', nn.LeakyReLU(0.2, inplace=True))]))

    def forward(self, feature):
        return self.layer(feature)


class ConvBlock(nn.Module):
    """
    Combination of convs of same output dim.
    """
    def __init__(self, num, dim, input_dim):
        """
        Args:
            num (int): the num of convs
            dim (int): the output of all convs
            input_dim (int): the input dim
        """
        super(ConvBlock, self).__init__()
        layers = [('convlayer%d' % (i + 1),
                   ConvLayer(dim if i else input_dim, dim))
                  for i in range(num)]
        self.block = nn.Sequential(OrderedDict(layers))

    def forward(self, features):
        return self.block(features)


class ConvNullBlock(nn.Module):
    def __init__(self):
        super(ConvNullBlock, self).__init__()

    def forward(self, x):
        return x


class EncoderConvModule(nn.Module):
    def __init__(self,
                 in_dim,
                 num_layers=[2, 2, 2, 2],
                 dims=[16, 32, 64, 128]):
        super(EncoderConvModule, self).__init__()
        self.module = nn.Sequential(
            OrderedDict([('convlayer0', ConvLayer(in_dim, dims[0]))]))
        for i in range(len(num_layers)):
            if num_layers[i]:
                block = ConvBlock(num_layers[i], dims[i], input_dim=dims[i])
                self.module.add_module('convblock%d' % (i + 1), block)
            if i != len(num_layers) - 1:
                down = nn.Conv2d(dims[i], dims[i + 1], 3, stride=2, padding=1)
                self.module.add_module('dowm%d' % (i + 1), down)
                relu = nn.LeakyReLU(0.2, inplace=True)
                self.module.add_module('drelu%d' % (i + 1), relu)

    def forward(self, x):
        return self.module(x)




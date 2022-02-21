import functools

import torch.nn as nn


class EncoderLayer(nn.Module):
    def __init__(self, input_nc, inner_nc, idx_layer, num_layers,
                 norm_layer=nn.BatchNorm2d):
        super(EncoderLayer, self).__init__()
        # Parameters.
        self.idx_layer = idx_layer
        self.num_layers = num_layers
        # Use bias for conv layer when norm_layer is InstanceNorm.
        if type(norm_layer) == functools.partial:
            use_bias = (norm_layer.func == nn.InstanceNorm2d)
        else:
            use_bias = (norm_layer == nn.InstanceNorm2d)
        # Sub layers of conv.
        self.relu = nn.LeakyReLU(0.2, True)
        self.conv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)  # o=floor(i+2p-k/s)+1
        self.norm = norm_layer(inner_nc)

    def forward(self, x):
        output = x
        if self.idx_layer == 0:
            # print("e%1d" % self.idx_layer)
            # Encoder的输入不用relu激活也不用norm.
            output = self.conv(output)
        elif self.idx_layer == self.num_layers - 1:
            # print("e%1d" % self.idx_layer)
            output = self.relu(output)
            output = self.conv(output)
            # Encoder的输出不用norm.
        else:
            # print("e%1d" % self.idx_layer)
            output = self.relu(output)
            output = self.conv(output)
            output = self.norm(output)
        return output


class Encoder(nn.Module):
    """A stack of 8 EncoderLayers, performs down-sampling."""

    def __init__(self, input_nc=1, ngf=64, num_layers=8,
                 norm_layer=nn.BatchNorm2d):
        super(Encoder, self).__init__()
        # Layer stack.
        self.layer_stack = nn.ModuleList()
        for idx_layer in range(num_layers):
            _input_nc = input_nc  # default: 1 or 3
            if idx_layer < 4:  # [0, 1, 2, 3]
                if idx_layer != 0:
                    _input_nc = ngf * pow(2, idx_layer - 1)
                inner_nc = ngf * pow(2, idx_layer)
            else:              # [4, 5, 6, 7]
                _input_nc = ngf * pow(2, 3)
                inner_nc = ngf * pow(2, 3)
            # Add encoder layer to the stack.
            self.layer_stack.append(EncoderLayer(input_nc=_input_nc, inner_nc=inner_nc,
                                                 idx_layer=idx_layer, num_layers=num_layers, norm_layer=norm_layer))
            # _input_nc: [1 or 3,  ngf * 1, ngf * 2, ngf * 4, ngf * 8, ngf * 8, ngf * 8, ngf * 8]
            #  inner_nc: [ngf * 1, ngf * 2, ngf * 4, ngf * 8, ngf * 8, ngf * 8, ngf * 8, ngf * 8]

    def forward(self, x):
        enc_outputs = []
        for encoder_layer in self.layer_stack:
            x = encoder_layer(x)
            enc_outputs.append(x)
        return enc_outputs

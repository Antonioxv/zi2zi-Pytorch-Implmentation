import functools

import torch
import torch.nn as nn


class DecoderLayer(nn.Module):
    def __init__(self, inner_nc, outer_nc, idx_layer, num_layers,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(DecoderLayer, self).__init__()
        # Parameters.
        self.idx_layer = idx_layer
        self.num_layers = num_layers
        # Use bias for conv layer when norm_layer is InstanceNorm2d.
        if type(norm_layer) == functools.partial:
            use_bias = (norm_layer.func == nn.InstanceNorm2d)
        else:
            use_bias = (norm_layer == nn.InstanceNorm2d)
        self.use_dropout = use_dropout
        # Sub layers of conv.
        self.relu = nn.ReLU(True)
        if idx_layer == num_layers - 1:
            self.conv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1)  # no bias
        else:
            self.conv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)  # o/=floor(i+2p-k/s)+1
        self.norm = norm_layer(outer_nc)
        self.af = nn.Tanh()  # Activation function.
        if use_dropout:
            self.dropout = nn.Dropout(p=0.5)

    def forward(self, enc_output, dec_output, cate=None):
        """
        The difference between tf implementation and mine is that
         the tensor concat is used for generate the (i+1)th decode_layer'input rather than (i)th decode_layer's output;
        So, the concat([cate, enc_output], 1) is part of the 1st decoder_layer,
         and the output linear projection is part of the last decoder_layer in my pytorch implementation, it simplifies the code a lot.
        """
        if self.idx_layer == 0:
            # print("d%1d" % self.idx_layer)
            # 1) Flatten the cate tensor from [40, 128] to [40, 128, 1, 1],
            # there is no need to do repeat(), cause we just add 2 pairs of parentheses.
            # 2) Concat the cate and enc_output where dim = 1.
            output = self.relu(torch.cat([cate.view(cate.shape[0], cate.shape[1], 1, 1), enc_output], dim=1))
            output = self.conv(output)
            output = self.norm(output)
        elif self.idx_layer == self.num_layers - 1:
            # print("d%1d" % self.idx_layer)
            # 1) Concat the enc_output and dec_output where dim = 1.
            output = self.relu(torch.cat([enc_output, dec_output], dim=1))
            output = self.conv(output)
            # 2) Decoder的输出不用norm.
            output = self.af(output)  # Activate the decoder's output, rescale to (-1, 1).
        else:
            # print("d%1d" % self.idx_layer)
            output = self.relu(torch.cat([enc_output, dec_output], dim=1))
            output = self.conv(output)
            output = self.norm(output)
            if self.use_dropout:
                output = self.dropout(output)  # Random dropout some parameters to avoid over-fitting.
        return output


class Decoder(nn.Module):
    """A stack of 8 DecoderLayers, performs up-sampling."""

    def __init__(self, output_nc=1, ngf=64, num_layers=8, embedding_dim=128,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(Decoder, self).__init__()
        # Parameters.
        self.num_layers = num_layers
        # Layer stack.
        self.layer_stack = nn.ModuleList()
        for idx_layer in range(num_layers):
            inner_nc = ngf * pow(2, 3) + embedding_dim  # default: ngf * 8 + 128
            outer_nc = output_nc                                  # default: 1 or 3
            if idx_layer < 4:  # [0, 1, 2, 3]
                if idx_layer != 0:
                    inner_nc = ngf * pow(2, 3) * 2  # Concat.
                outer_nc = ngf * pow(2, 3)
            else:              # [4, 5, 6, 7]
                inner_nc = ngf * pow(2, (num_layers - 1) - idx_layer) * 2  # Concat.
                if idx_layer != num_layers - 1:
                    outer_nc = ngf * pow(2, (num_layers - 1) - idx_layer - 1)
            # Add decoder layer to the stack.
            self.layer_stack.append(DecoderLayer(inner_nc=inner_nc, outer_nc=outer_nc,
                                                 idx_layer=idx_layer, num_layers=num_layers, norm_layer=norm_layer,
                                                 use_dropout=use_dropout))
            # inner_nc: [ngf * 8 + embedding_dim, ngf * 16, ngf * 16, ngf * 16, ngf * 16, ngf * 8, ngf * 4, ngf * 2]
            # outer_nc: [ngf * 8,                 ngf * 8,  ngf * 8,  ngf * 8,  ngf * 4,  ngf * 2, ngf * 1, 1 or 3]

    def forward(self, enc_outputs, cate):
        dec_output = None
        for decoder_layer, idx in zip(self.layer_stack, range(self.num_layers)):
            if idx == 0:
                dec_output = decoder_layer(enc_outputs[(self.num_layers - 1) - idx], dec_output, cate)
            else:
                dec_output = decoder_layer(enc_outputs[(self.num_layers - 1) - idx], dec_output)
        return dec_output

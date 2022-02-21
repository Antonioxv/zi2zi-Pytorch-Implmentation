import math
import functools

import torch.nn as nn


class Discriminator(nn.Module):
    """A discriminator based on patch-GAN."""

    def __init__(self, input_nc=3*2, image_size=256, ndf=64, embedding_num=40,
                 norm_layer=nn.BatchNorm2d):
        """
            Construct the discriminator.
        :param input_nc:             num of channels in discriminator's input;
        :param image_size:           height and width of the images, here is 256;
        :param ndf:                  the number of output channels in the generator's first conv layer;
        :param embedding_num:        num of distinct types of fonts;
        :param norm_layer:           normalization layer, usually BatchNorm2d.
        """
        super(Discriminator, self).__init__()
        # 1) Parameters.
        # No need to use bias as BatchNorm2d has affine parameters
        if type(norm_layer) == functools.partial:
            use_bias = (norm_layer.func != nn.BatchNorm2d)
        else:
            use_bias = (norm_layer != nn.BatchNorm2d)
        self.layer_stack = nn.ModuleList()

        # 2) Layers.
        # The tf implementation' kernel_size = 5, and it use "SAME" padding, so the padding: p = floor(k / 2) = 2;
        k = 5
        p = int(math.floor(k / 2))  # 2
        # In tf implementation, there are only 3 conv2d layers with stride = 2, h0, h1 and h2
        # h0
        self.layer_stack.append(nn.Conv2d(input_nc, ndf, kernel_size=k, stride=2, padding=p))
        self.layer_stack.append(nn.LeakyReLU(0.2, True))

        nf_mult_prev = 1
        nf_mult = 1
        # h1, h2
        for n in range(1, 3):
            nf_mult_prev = nf_mult
            nf_mult = pow(2, n)
            self.layer_stack.append(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=k, stride=2, padding=p, bias=use_bias))
            self.layer_stack.append(norm_layer(ndf * nf_mult))
            self.layer_stack.append(nn.LeakyReLU(0.2, True))

        nf_mult_prev = nf_mult  # 4
        nf_mult = 8
        # h3
        self.layer_stack.append(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=5, stride=1, padding=2, bias=use_bias))
        self.layer_stack.append(norm_layer(ndf * nf_mult))
        self.layer_stack.append(nn.LeakyReLU(0.2, True))

        # # Maybe useful? Experiment need to be done later.
        # # Output 1 channel prediction map
        # self.layer_stack.append(nn.Conv2d(ndf * nf_mult, 1, kernel_size=5, stride=1, padding=2))

        # 3) Calculate num of features and init linear projections.
        final_channels = ndf * nf_mult  # 512
        # o = floor((i + 2 * p - k) / s) + 1, as i = 256, p = 2, k = 5, s = 2
        # h0, h1 and h2
        for n in range(0, 3):
            image_size = int(math.floor((image_size + 2 * p - k) / 2)) + 1
        # h3
        image_size = int(math.floor((image_size + 2 * p - k) / 1)) + 1  # image_size = w = h = 32
        # 524288 = 512 * w * h = 2^19 (input_w = input_h = 256)
        # 131072 = 512 * w * h = 2^17 (input_w = input_h = 128)
        final_features = final_channels * image_size * image_size

        # Linear projections, for calculating bin and category logits
        self.binary = nn.Linear(final_features, 1)
        self.category = nn.Linear(final_features, embedding_num)

    def forward(self, x):
        """
        Standard forward, [bs, 2, 256, 256] -> [bs, 64, 128, 128] -> [bs, 128, 64, 64] -> [bs, 256, 32, 32] -> [bs, 512, 32, 32].
        :param x:   images, shape: [batch_size, input_nc, 256, 256]
        :return:    binary_logits, category_logits
        """
        batch_size = x.shape[0]
        # 1) Layers.
        for layer in self.layer_stack:
            x = layer(x)
        # 2) Reshape features to [batch_size, 524288]
        features = x.view(batch_size, -1)
        # 3) Linear projections.
        binary_logits = self.binary(features)
        category_logits = self.category(features)
        return binary_logits, category_logits

import torch.nn as nn

from .decoder import Decoder
from .encoder import Encoder


class UNetGenerator(nn.Module):
    """A generator based on Unet."""

    def __init__(self, input_nc=3, output_nc=3, num_downs=8, ngf=64, embedding_num=40, embedding_dim=128,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        """
            Construct the generator.
        :param input_nc:        the number of channels in input images;
        :param output_nc:       the number of channels in output images;
        :param num_downs:       the number of down-samplings in UNet. For example, if |num_downs| == 8,
                                image of size 256x256 will become of size 1x1 # at the bottleneck;
        :param ngf:             the number of output channels in the generator's first conv layer;
        :param embedding_num:   the number of category embedding;
        :param embedding_dim:   the dimension of category embedding;
        :param norm_layer:      normalization layer, usually BatchNorm2d;
        :param use_dropout:     use dropout or not.
        """
        super(UNetGenerator, self).__init__()
        # 1) Parameters.
        self.num_downs = num_downs
        # 2) Construct the encoder and decoder according to the U-net architecture.
        self.encoder = Encoder(
            input_nc=input_nc, ngf=ngf, num_layers=num_downs,
            norm_layer=norm_layer
        )
        self.decoder = Decoder(
            output_nc=output_nc, ngf=ngf, num_layers=num_downs, embedding_dim=embedding_dim,
            norm_layer=norm_layer, use_dropout=use_dropout
        )
        # 3) Category embedding.
        self.cate_emb = nn.Embedding(embedding_num, embedding_dim)  # shape: [40, 128]

    def forward(self, x, label=None):
        """Standard forward."""
        enc_outputs = self.encoder(x)
        # if label is not None and 'LongTensor' in label.type():
        if label is not None:
            cate = self.cate_emb(label)
            return enc_outputs[self.num_downs - 1], self.decoder(enc_outputs, cate)  # Both 2.
        else:
            return enc_outputs[self.num_downs - 1]  # Encoded image, shape: [bs, ngf * 8, 1, 1]

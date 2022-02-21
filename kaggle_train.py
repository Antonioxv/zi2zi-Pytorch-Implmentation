import os
import math
import time
import torch
import random
import pickle
import functools

import numpy as np
import torch.nn as nn
import torch.utils.data as data
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torchvision.transforms as transforms

from io import BytesIO
from torch.nn import init
from PIL import Image, ImageFilter
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader


'''utils'''


def bytes_to_file(bytes_img):
    return BytesIO(bytes_img)


class PickledImageProvider(object):
    def __init__(self, obj_path):
        self.obj_path = obj_path
        self.examples = self.load_pickled_examples()

    def load_pickled_examples(self):
        with open(self.obj_path, "rb") as of:
            examples = list()
            while True:
                try:
                    e = pickle.load(of)
                    examples.append(e)
                    if len(examples) % 100000 == 0:
                        print("processed %d examples" % len(examples))
                except EOFError:
                    break
                except Exception:
                    pass
            print("unpickled total %d examples" % len(examples))
            return examples


def read_split_image(img):
    box1 = (0, 0, img.size[1], img.size[1])  # (left, upper, right, lower) - tuple
    box2 = (img.size[1], 0, img.size[0], img.size[1])
    img_A = img.crop(box1)  # target
    img_B = img.crop(box2)  # source
    return img_A, img_B


def plot_tensor(tensor):
    img = np.transpose(tensor.data, (1, 2, 0))
    plt.imshow(img)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=None):
    """
    Initialize a network:
    1. register CPU/GPU device (with multi-GPU support);
    2. initialize the network weights.
    :param net: the network to be initialized;
    :param init_type: the name of an initialization method: normal | xavier | kaiming | orthogonal;
    :param init_gain: scaling factor for normal, xavier and orthogonal;
    :param gpu_ids: which GPUs the network runs on: e.g., 0,1,2;
    :return: an initialized network.
    """
    if gpu_ids:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        # net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def init_weights(net, init_type='normal', init_gain=0.02):
    """
    Initialize network weights.
    'Normal' is used in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    :param net: network to be initialized;
    :param init_type: the name of an initialization method: normal | xavier | kaiming | orthogonal;
    :param init_gain: scaling factor for normal, xavier and orthogonal.
    :return: an initialized network.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


'''data'''


class DatasetFromObj(data.Dataset):
    """ Override the standard torch Dataset class. """
    def __init__(self, obj_path, input_nc, augment=False, bold=False, rotate=False, blur=False, start_from=0):
        super(DatasetFromObj, self).__init__()
        self.image_provider = PickledImageProvider(obj_path)
        self.input_nc = input_nc
        if self.input_nc == 1:
            # print("input_nc==1")
            self.transform = transforms.Compose([
                # transforms.Grayscale(num_output_channels=1),  # 彩色图像转灰度图像num_output_channels默认1
                transforms.ToTensor(),
                transforms.Normalize(mean=0.5, std=0.5)
            ])
        elif self.input_nc == 3:
            # print("input_nc==3")
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])
        else:
            raise ValueError('input_nc should be 1 or 3')

        self.augment = augment
        self.bold = bold
        self.rotate = rotate
        self.blur = blur
        self.start_from = start_from

    def __getitem__(self, index):
        item = self.image_provider.examples[index]
        img_A, img_B = self.process(item[1])
        return item[0] - self.start_from, img_A, img_B

    def __len__(self):
        return len(self.image_provider.examples)

    def process(self, img_bytes):
        """
        byte stream to training data entry
        """
        image_file = bytes_to_file(img_bytes)
        img = Image.open(image_file)
        # modify
        if self.input_nc == 1:
            img = img.convert('L')
        try:
            img_A, img_B = read_split_image(img)  # tgt_img, src_img
            if self.augment:
                # augment the image by:
                # 1) enlarge the image
                # 2) random crop the image back to its original size
                # NOTE: image A and B needs to be in sync as how much
                # to be shifted
                w, h = img_A.size

                # Generate bold imgs.
                if self.bold:
                    multiplier = random.uniform(1.0, 1.2)
                else:
                    multiplier = random.uniform(1.0, 1.05)

                # add an eps to prevent cropping issue
                nw = int(multiplier * w) + 1
                nh = int(multiplier * h) + 1

                # Used to use Image.BICUBIC, change to ANTIALIAS, get better image.
                img_A = img_A.resize((nw, nh), Image.ANTIALIAS)
                img_B = img_B.resize((nw, nh), Image.ANTIALIAS)

                shift_x = random.randint(0, max(nw - w - 1, 0))
                shift_y = random.randint(0, max(nh - h - 1, 0))

                img_A = img_A.crop((shift_x, shift_y, shift_x + w, shift_y + h))
                img_B = img_B.crop((shift_x, shift_y, shift_x + w, shift_y + h))

                # Rotate imgs.
                if self.rotate and random.random() > 0.9:
                    angle_list = [0, 180]
                    random_angle = random.choice(angle_list)
                    if self.input_nc == 3:
                        fill_color = (255, 255, 255)
                    else:
                        fill_color = 255
                    img_A = img_A.rotate(random_angle, resample=Image.BILINEAR, fillcolor=fill_color)
                    img_B = img_B.rotate(random_angle, resample=Image.BILINEAR, fillcolor=fill_color)

                # Generate blurry imgs.
                if self.blur and random.random() > 0.8:
                    sigma_list = [1, 1.5, 2]
                    sigma = random.choice(sigma_list)
                    img_A = img_A.filter(ImageFilter.GaussianBlur(radius=sigma))
                    img_B = img_B.filter(ImageFilter.GaussianBlur(radius=sigma))

                '''
                Used to resize here. Change it before rotate and blur.
                w_offset = random.randint(0, max(0, nh - h - 1))
                h_offset = random.randint(0, max(0, nh - h - 1))
                img_A = img_A[:, h_offset: h_offset + h, w_offset: w_offset + h]
                img_B = img_B[:, h_offset: h_offset + h, w_offset: w_offset + h]
                '''

                img_A = self.transform(img_A)
                img_B = self.transform(img_B)
            else:
                img_A = self.transform(img_A)
                img_B = self.transform(img_B)

            return img_A, img_B
        finally:
            # process done
            image_file.close()


'''model'''



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
        if label is not None and 'LongTensor' in label.type():
            cate = self.cate_emb(label)
            return enc_outputs[self.num_downs - 1], self.decoder(enc_outputs, cate)  # Both 2.
        else:
            return enc_outputs[self.num_downs - 1]  # Encoded image, shape: [B, ngf * 8, 1, 1]


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


class CategoryLoss(nn.Module):
    def __init__(self, category_num):
        super(CategoryLoss, self).__init__()
        emb = nn.Embedding(category_num, category_num)
        emb.weight.data = torch.eye(category_num)
        self.emb = emb
        self.loss = nn.BCEWithLogitsLoss()  # bce

    def forward(self, category_logits, labels):
        target = self.emb(labels)
        return self.loss(category_logits, target)


class BinaryLoss(nn.Module):
    def __init__(self):
        super(BinaryLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, real=True):
        if real:
            labels = torch.ones(logits.shape[0], 1)
        else:
            labels = torch.zeros(logits.shape[0], 1)
        if logits.is_cuda:  # move to gpu when gpu is available
            labels = labels.cuda()
        return self.bce(logits, labels)


class Zi2Zi:
    """Pytorch implementation of zi2zi, author: Chaoxin FU."""
    def __init__(self, input_nc=1, image_size=256,
                 embedding_num=40, embedding_dim=128, ngf=64, ndf=64,
                 Lconst_penalty=15, Lcategory_penalty=1, L1_penalty=100,  # Ltv_penalty=0
                 lr=0.001, gpu_ids=None, save_dir='.', is_training=True):
        """
        Initialize zi2zi model, including save parameters and init nn, optimizers and losses.
        :param input_nc:                the number of channels in input images;
        :param image_size:              height and width of input images;
        :param embedding_num:           the number of category embedding;
        :param embedding_dim:           the dimension of category embedding;
        :param ngf:                     the number of output channels in the generator's first conv layer;
        :param ndf:                     the number of output channels in the generator's first conv layer;
        :param Lconst_penalty:          constant;
        :param Lcategory_penalty:       constant;
        :param L1_penalty:              constant;
        :param lr:
        :param gpu_ids:
        :param save_dir:
        :param is_training:
        """

        '''Save local parameters.'''
        self.input_nc = input_nc
        self.image_size = image_size

        self.embedding_num = embedding_num
        self.embedding_dim = embedding_dim
        self.ngf = ngf
        self.ndf = ndf

        self.Lconst_penalty = Lconst_penalty
        self.Lcategory_penalty = Lcategory_penalty
        self.L1_penalty = L1_penalty
        # self.Ltv_penalty = Ltv_penalty

        self.lr = lr
        self.gpu_ids = gpu_ids
        self.save_dir = save_dir

        # When training, random dropout some parameters to avoid overfitting.
        self.is_training = is_training
        if self.is_training:
            self.use_dropout = True
        else:
            self.use_dropout = False

        '''Init neural network UNetGenerator and Discriminator.'''
        # Init generator and discriminator.
        self.netG = UNetGenerator(
            input_nc=self.input_nc,
            output_nc=self.input_nc,
            num_downs=int(math.log2(image_size)),
            ngf=self.ngf,
            embedding_num=self.embedding_num,
            embedding_dim=self.embedding_dim,
            use_dropout=self.use_dropout
        )
        self.netD = Discriminator(
            input_nc=self.input_nc * 2,
            image_size=self.image_size,
            ndf=self.ndf,
            embedding_num=self.embedding_num
        )

        # Register CPU/GPU device and init weights.
        init_net(self.netG, gpu_ids=self.gpu_ids)
        init_net(self.netD, gpu_ids=self.gpu_ids)

        '''Init optimizers for Generator and Discriminator.'''
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=self.lr, betas=(0.5, 0.999))

        '''Init losses and other functions.'''
        self.category_loss = CategoryLoss(self.embedding_num)
        self.binary_loss = BinaryLoss()
        self.l1_loss = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.sigmoid = nn.Sigmoid()

    def setup(self):
        # Move parameters to gpu when gpu is available.
        if self.gpu_ids is not None:
            self.category_loss.cuda()
            self.binary_loss.cuda()
            self.l1_loss.cuda()
            self.mse.cuda()
            self.sigmoid.cuda()

        # Train or eval.
        if self.is_training:
            self.netG.train()
            self.netD.train()
        else:
            self.netG.eval()
            self.netD.eval()

    def set_input(self, labels, real_A, real_B):  # [src_img, tgt_img]
        # Move parameters to gpu when gpu is available.
        if self.gpu_ids is not None:
            self.labels = labels.to(self.gpu_ids[0])
            self.real_A = real_A.to(self.gpu_ids[0])
            self.real_B = real_B.to(self.gpu_ids[0])
        else:
            self.labels = labels
            self.real_A = real_A
            self.real_B = real_B

    def update_lr(self):
        # Update lr for optimizer_D
        for p in self.optimizer_D.param_groups:
            current_lr = p['lr']
            updated_lr = current_lr / 2.0
            # Minimum learning rate guarantee
            updated_lr = max(updated_lr, 0.0002)
            p['lr'] = updated_lr
            print("Decay net_D learning rate from %.5f to %.5f." % (current_lr, updated_lr))
        # Update lr for optimizer_G
        for p in self.optimizer_G.param_groups:
            current_lr = p['lr']
            updated_lr = current_lr / 2.0
            # minimum learning rate guarantee
            updated_lr = max(updated_lr, 0.0002)
            p['lr'] = updated_lr
            print("Decay net_G learning rate from %.5f to %.5f." % (current_lr, updated_lr))

    def generate(self):
        # 1) Encode real_A, and generate fake_B;
        encoded_real_A, self.fake_B = self.netG(self.real_A, self.labels)
        self.encoded_real_A = encoded_real_A.view(self.fake_B.shape[0], -1)  # [bs, 512, 1, 1] -> [bs, 512]
        # 2) Encode fake_B
        self.encoded_fake_B = self.netG(self.fake_B).view(self.fake_B.shape[0], -1)  # [bs, 512] -> [bs, 512]

    def backward_D(self, no_target_source=False):
        # 1) Concat images.
        real_AB = torch.cat([self.real_A, self.real_B], 1)  # [src_img, tgt_img]
        fake_AB = torch.cat([self.real_A, self.fake_B], 1)

        # 2) Discriminate, and get bin_logits and category_logits for both fake and real images.
        real_D_logits, real_category_logits = self.netD(real_AB)
        fake_D_logits, fake_category_logits = self.netD(fake_AB.detach())

        # 3) Use category_loss to calculate category losses for both fake and real images.
        real_category_loss = self.category_loss(real_category_logits, self.labels)
        fake_category_loss = self.category_loss(fake_category_logits, self.labels)
        category_loss = (real_category_loss + fake_category_loss) * self.Lcategory_penalty  # total category loss

        # 4) Use binary_loss to calculate bin losses for both fake and real images.
        d_loss_real = self.binary_loss(real_D_logits, real=True)   # compared with all-ones!
        d_loss_fake = self.binary_loss(fake_D_logits, real=False)  # compared with all-zeroes!

        # 5) Calculate total loss for D.
        self.d_loss = (d_loss_real + d_loss_fake) + category_loss / 2.0
        self.d_loss.backward()  # backpropagation for D
        return category_loss

    def backward_G(self, no_target_source=False):
        # 1) Concat images generated by G.
        fake_AB = torch.cat([self.real_A, self.fake_B], 1)

        # 2) Discriminate, and get bin_logits and category_logits for only fake images.
        fake_D_logits, fake_category_logits = self.netD(fake_AB)

        # 3) Use binary_loss to calculate cheat loss.
        cheat_loss = self.binary_loss(fake_D_logits, real=True)  # compared with all-ones!

        # 4) Use category_loss to calculate category losses for only fake images.
        fake_category_loss = self.category_loss(fake_category_logits, self.labels) * self.Lcategory_penalty

        # 5) L1 loss between real and generated images
        l1_loss = self.l1_loss(self.fake_B, self.real_B) * self.L1_penalty

        # 6) Calculate encoding constant loss,
        # this loss assume that generated imaged and real image should reside in the same space and close to each other.
        const_loss = self.mse(self.encoded_real_A, self.encoded_fake_B) * self.Lconst_penalty

        # 7) Calculate total loss for G.
        self.g_loss = cheat_loss + fake_category_loss + const_loss + l1_loss
        self.g_loss.backward()  # backpropagation for G
        return cheat_loss, l1_loss, const_loss

    def forward(self):
        # 1) Generate fake images G(B), encoded real images E(A) and encoded fake images E(G(B))
        self.generate()

        # 2) Update D
        self.set_requires_grad(self.netD, True)  # enable backpropagation for D
        self.optimizer_D.zero_grad()  # forward, and set D's gradients to zero
        category_loss = self.backward_D()  # calculate losses, and do backpropagation for D
        self.optimizer_D.step()  # update D's weights

        # 3) Update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # forward, and set G's gradients to zero
        self.backward_G()  # calculate losses, and do backpropagation for G
        self.optimizer_G.step()  # updpate G's weights

        # 4) Magic move to optimize G again,
        # according to https://github.com/carpedm20/DCGAN-tensorflow,
        # collect all the losses along the way.
        self.generate()  # generate
        self.optimizer_G.zero_grad()  # forward
        cheat_loss, l1_loss, const_loss = self.backward_G()  # backward
        self.optimizer_G.step()  # update G's weights

        return category_loss, const_loss, l1_loss, cheat_loss

    def set_requires_grad(self, nets, requires_grad=False):
        """
            Set requies_grad=False for all the networks to avoid unnecessary computations.
        :param nets:                a list of networks;
        :param requires_grad:       whether the networks require gradients or not.
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def print_networks(self, verbose=False):
        """
            Print the total number of parameters in the network and (if verbose) network architecture.
        :param verbose:     if verbose: print the network architecture
        :return:
        """
        print('------------- Networks initialized -------------')
        for name in ['G', 'D']:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def save_networks(self, epoch):
        """
            Save all the networks to the disk.
        :param epoch:       current epoch, is used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in ['G', 'D']:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if self.gpu_ids and torch.cuda.is_available():
                    # torch.save(net.cpu().state_dict(), save_path)
                    torch.save(net.state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def load_networks(self, epoch):
        """
            Load all the networks from the disk.
        :param epoch:       current epoch, used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in ['G', 'D']:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)

                if self.gpu_ids and torch.cuda.is_available():
                    net.load_state_dict(torch.load(load_path))
                else:
                    net.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')))
        print('load model %d' % epoch)

    def sample(self, batch, basename):
        """
            Sample images from train.obj or val.obj, or a specified text in specified font.
        :param batch:           bach data from data-loader;
        :param basename:        path to save sample images.
        :return:
        """
        os.makedirs(basename, exist_ok=True)
        cnt = 0
        with torch.no_grad():
            self.set_input(batch[0], batch[2], batch[1])
            self.generate()
            tensor_to_plot = torch.cat([self.fake_B, self.real_B], 3)
            for label, image_tensor in zip(batch[0], tensor_to_plot):
                label_dir = os.path.join(basename, str(label.item()))
                os.makedirs(label_dir, exist_ok=True)
                vutils.save_image(image_tensor, os.path.join(label_dir, str(cnt) + '.png'))
                cnt += 1
            # img = vutils.make_grid(tensor_to_plot)
            # vutils.save_image(tensor_to_plot, basename + "_construct.png")
            '''
            We don't need generate_img currently.
            self.set_input(torch.randn(1, self.embedding_num).repeat(batch[0].shape[0], 1), batch[2], batch[1])
            self.generate()
            tensor_to_plot = torch.cat([self.fake_B, self.real_A], 3)
            vutils.save_image(tensor_to_plot, basename + "_generate.png")
            '''


'''train'''


def main():
    print(torch.__version__)
    print(torch.cuda.is_available())

    image_size = 256
    batch_size = 32
    epoch = 201
    lr = 0.01
    schedule = 20
    sample_steps = 400
    checkpoint_steps = 1000
    random_seed = 777
    # gpu_ids = None
    gpu_ids = [0]
    input_nc = 1  # instead of 3
    L1_penalty = 100
    Lconst_penalty = 15
    # Ltv_penalty = 0.0
    Lcategory_penalty = 1
    embedding_num = 40
    embedding_dim = 128
    resume = False

    ''' Fix random seed during the experiment. '''
    random.seed(random_seed)  # random seed is 777
    torch.manual_seed(random_seed)

    ''' Make experiment dirs. '''
    # experiment_dir = 'experiment_8'                     # local experiment dir
    # os.makedirs(experiment_dir, exist_ok=True)
    experiment_dir = 'logs/experiment_8'              # kaggle experiment dir
    # data_dir = os.path.join(experiment_dir, "data")     # local data dir
    # os.makedirs(data_dir, exist_ok=True)
    data_dir = '../input/gan-dataset6'                # kaggle data dir
    # Copy obj to data path
    # os.
    checkpoint_dir = os.path.join(experiment_dir, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)
    sample_dir = os.path.join(experiment_dir, "sample")
    os.makedirs(sample_dir, exist_ok=True)
    # Tensorboard
    run_dir = os.path.join(experiment_dir, "runs")
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(run_dir)

    ''' Init the zi2zi model. '''
    # Device: CPU/GPU
    gpu_ids = gpu_ids if torch.cuda.is_available() else None
    # Zi2Zi does not inherit nn.Module, but Gen and Dis do, so it can be encapsulated as a nn.Module-like class.
    model = Zi2Zi(
        input_nc=input_nc,
        image_size=image_size,
        embedding_num=embedding_num,
        embedding_dim=embedding_dim,
        L1_penalty=L1_penalty,
        Lconst_penalty=Lconst_penalty,
        Lcategory_penalty=Lcategory_penalty,
        # Ltv_penalty=Ltv_penalty,
        lr=lr,
        gpu_ids=gpu_ids,
        save_dir=checkpoint_dir,
        is_training=True
    )
    print('Model initialized.')
    # Continue training from resume step.
    if resume:
        model.load_networks(resume)
    # Move parameters to gpu when gpu is available, and set nn.Module train or eval.
    model.setup()
    # Print model' nn architecture.
    model.print_networks(True)

    ''' Process datasets and data loaders. '''
    # Train dataset and dataloader.
    train_obj = os.path.join(data_dir, 'train.obj')
    # train_dataset = DatasetFromObj(train_obj, input_nc=input_nc, augment=True, bold=False, rotate=False, blur=True)
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # See more in train part.
    # Validate dataset and dataloader.
    val_obj = os.path.join(data_dir, 'val.obj')
    val_dataset = DatasetFromObj(val_obj, input_nc=input_nc)  # No augment.
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # Val dataset load only once, no shuffle.

    ''' Train. '''
    global_steps = 0
    start_time = time.time()
    for epoch in range(epoch):
        # 1) Generate train dataset every epoch, so that different styles of saved char images can be trained.
        # No bold and no rotate, only generate blur, the first 2 is not performing well in the dataset!
        train_dataset = DatasetFromObj(train_obj, input_nc=input_nc, augment=True, bold=False, rotate=False, blur=True)
        # Shuffling.
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # Cal once is enough.
        num_batch = math.ceil(len(train_dataset) / batch_size)  # Cal once is enough.

        # 2) Train one epoch
        for bid, batch in enumerate(train_dataloader):  # [bid, [labels, tgt_imgs, src_imgs]]
            # 2.1) Set input data and train one step.
            model.set_input(batch[0], batch[2], batch[1])  # [labels, src_imgs, tgt_imgs]
            category_loss, const_loss, l1_loss, cheat_loss = model.forward()  # nn.Module-like class

            # 2.2) Print log and save it to tensorboard.
            if bid % 10 == 0:
                writer.add_scalar('Train/g_loss', model.g_loss.item(), global_step=global_steps)
                writer.add_scalar('Train/d_loss', model.d_loss.item(), global_step=global_steps)
            if bid % 100 == 0:
                passed_time = time.time() - start_time
                log_format = "Epoch: [%3d], [%4d/%4d] time: %4.2f, g_loss: %.5f, d_loss: %.5f, " + \
                             "category_loss: %.5f, cheat_loss: %.5f, l1_loss: %.5f, const_loss: %.5f"
                print(log_format % (epoch, bid, num_batch, passed_time, model.g_loss.item(), model.d_loss.item(),
                                    category_loss, cheat_loss, l1_loss, const_loss))
            # 2.3) Save checkpoint.
            if global_steps % checkpoint_steps == 0:
                model.save_networks(global_steps)
                print("Checkpoint: save checkpoint step %d" % global_steps)
            # 2.4) Sample images from val dataset.
            if global_steps % sample_steps == 0:
                for vbid, val_batch in enumerate(val_dataloader):
                    model.sample(val_batch, os.path.join(sample_dir, str(global_steps)))
                print("Sample: sample step %d" % global_steps)

            global_steps += 1

        # 3) Update learning rate.
        if (epoch + 1) % schedule == 0:
            model.update_lr()

    ''' Validate. '''
    for vbid, val_batch in enumerate(val_dataloader):
        model.sample(val_batch, os.path.join(sample_dir, str(global_steps)))

    # Save final version of checkpoint.
    model.save_networks(global_steps)
    print("Checkpoint: save checkpoint step %d" % global_steps)

    writer.close()
    print('Terminated')


if __name__ == '__main__':
    main()

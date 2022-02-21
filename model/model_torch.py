import math
import os

import torch
import torch.nn as nn
import torchvision.utils as vutils

from utils.init_net import init_net
from .generator import UNetGenerator
from .discriminator import Discriminator
from .losses import CategoryLoss, BinaryLoss


class Zi2Zi:
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
        :param lr:                      learning rate;
        :param gpu_ids:                 device, cpu or gpu;
        :param save_dir:                path to save model;
        :param is_training:             is traing or not.
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
        self.use_dropout = True

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
        self.encoded_fake_B = self.netG(self.fake_B).view(self.fake_B.shape[0], -1)  # [bs, 512， 1， 1] -> [bs, 512]

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
            # for label, image_tensor in zip(batch[0], tensor_to_plot):
            for label, idx in zip(batch[0], range(int(tensor_to_plot.shape[0] / 5))):
                tensor_to_plot_slice = tensor_to_plot[idx * 5: (idx + 1) * 5]
                label_dir = os.path.join(basename, str(label.item()))
                os.makedirs(label_dir, exist_ok=True)
                img = vutils.make_grid(tensor_to_plot_slice, padding=2)
                vutils.save_image(img, os.path.join(label_dir, str(idx) + '.png'))
                # vutils.save_image(image_tensor, os.path.join(label_dir, str(cnt) + '.png'))
                # cnt += 1

            '''
            We don't need generate_img currently.
            self.set_input(torch.randn(1, self.embedding_num).repeat(batch[0].shape[0], 1), batch[2], batch[1])
            self.generate()
            tensor_to_plot = torch.cat([self.fake_B, self.real_A], 3)
            vutils.save_image(tensor_to_plot, basename + "_generate.png")
            '''

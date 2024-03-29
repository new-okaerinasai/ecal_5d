import torch
from torch import nn
from .translation_generator import Generator
from .discriminator_wrapper import DiscriminatorWrapper
from .discriminator_loss import DiscriminatorLoss
from .perceptual_loss import FeatureExtractor
from itertools import chain
from src import utils
from torch.autograd import Variable
import os



class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()

        self.gpu_id = opt.gpu_ids[0]
        self.weights_path = os.path.join(opt.experiment_path, 'checkpoints')

        # Generators
        self.gen_A = Generator(opt, 'A', opt.gen_type_name_A)
        self.gen_B = Generator(opt, 'B', opt.gen_type_name_B)

        # Discriminators
        self.dis_A = DiscriminatorWrapper(opt, 'A')
        self.dis_B = DiscriminatorWrapper(opt, 'B')

        # Load weights
        utils.load_checkpoint(self, opt.which_epoch, opt.pretrained_gen_path)

        # Print architectures
        print('\nGen A to B\n')
        num_params = 0
        for p in self.gen_B.parameters():
            num_params += p.numel()
        print(self.gen_B)
        print('Number of parameters: %d' % num_params)

        print('\nGen B to A\n')
        num_params = 0
        for p in self.gen_A.parameters():
            num_params += p.numel()
        print(self.gen_A)
        print('Number of parameters: %d' % num_params)

        print('\nDis A\n')
        num_params = 0
        for p in self.dis_A.parameters():
            num_params += p.numel()
        print(self.dis_A)
        print('Number of parameters: %d' % num_params)

        print('\nDis B\n')
        num_params = 0
        for p in self.dis_B.parameters():
            num_params += p.numel()
        print(self.dis_B)
        print('Number of parameters: %d' % num_params)

        self.gen_params = chain(
            self.gen_A.parameters(),
            self.gen_B.parameters())

        self.dis_params = chain(
            self.dis_A.parameters(),
            self.dis_B.parameters())

        # Losses
        self.crit_dis_A = DiscriminatorLoss(opt, self.dis_A)
        self.crit_dis_B = DiscriminatorLoss(opt, self.dis_B)

        # If an encoder is required, load the weights
        if (opt.mse_loss_type_A == 'perceptual' or
            opt.mse_loss_type_B == 'perceptual' or
            hasattr(self, 'dis_A') and self.dis_A.use_encoder or 
            hasattr(self, 'dis_B') and self.dis_B.use_encoder):

            # Load encoder
            if opt.enc_type[:5] == 'vgg19':
                layers = '1,6,11,20,29'

            self.enc = FeatureExtractor(
                input_range='tanh',
                net_type=opt.enc_type,
                layers=layers).eval()

            print('')
            print(self.enc)
            print('')

        else:

            self.enc = None

        self.crit_mse_A = utils.get_criterion(
            opt.mse_loss_type_A, 
            opt.mse_loss_weight_A,
            self.enc)
        self.crit_mse_B = utils.get_criterion(
            opt.mse_loss_type_B, 
            opt.mse_loss_weight_B,
            self.enc)

        self.weights_path = os.path.join(opt.experiment_path, 'checkpoints')

        # In case domains have different sizes, this is needed for mse loss
        scale_factor = opt.img_size_B // opt.img_size_A

        self.down = nn.AvgPool2d(scale_factor)
        self.up = nn.Upsample(
            scale_factor=scale_factor, 
            mode='bilinear',
            align_corners=False)

        # Load onto gpus
        self.gen_A = nn.DataParallel(self.gen_A.cuda(self.gpu_id), opt.gpu_ids)
        self.gen_B = nn.DataParallel(self.gen_B.cuda(self.gpu_id), opt.gpu_ids)
        self.dis_A = nn.DataParallel(self.dis_A.cuda(self.gpu_id), opt.gpu_ids)
        self.dis_B = nn.DataParallel(self.dis_B.cuda(self.gpu_id), opt.gpu_ids)
        if self.enc is not None: 
            self.enc = nn.DataParallel(self.enc.cuda(self.gpu_id), opt.gpu_ids)

    def forward(self, inputs):

        if len(inputs) == 2:
            real_A, real_B = inputs
        elif len(inputs) == 4:
            real_A, real_A_aux, real_B, real_B_aux = inputs

        # Input images
        self.real_A = Variable(real_A.cuda(self.gpu_id))
        self.real_B = Variable(real_B.cuda(self.gpu_id))

        # Fake images
        self.fake_B = self.gen_B(self.real_A)
        self.fake_A = self.gen_A(self.real_B)

    def backward_G(self):

        # Cycle loss
        cycle_A = self.gen_A(self.fake_B)
        cycle_B = self.gen_B(self.fake_A)

        self.loss_cycle_A = self.crit_mse_A(cycle_A, self.real_A)
        self.loss_cycle_B = self.crit_mse_B(cycle_B, self.real_B)

        # Identity loss
        ident_A = self.gen_A(self.up(self.real_A))
        ident_B = self.gen_B(self.down(self.real_B))

        self.loss_ident_A = self.crit_mse_A(ident_A, self.real_A)
        self.loss_ident_B = self.crit_mse_B(ident_B, self.real_B)

        # MSE loss
        loss_mse_A = self.loss_cycle_A + self.loss_ident_A
        loss_mse_B = self.loss_cycle_B + self.loss_ident_B

        # GAN loss
        loss_dis_A, _, _ = self.crit_dis_A(
            img_real_dst=self.fake_A,
            enc=self.enc)
        loss_dis_B, _, _ = self.crit_dis_B(
            img_real_dst=self.fake_B,
            enc=self.enc)

        loss_G = loss_mse_A + loss_mse_B + loss_dis_A + loss_dis_B

        if self.training:
            loss_G.backward()

        # Get values for visualization
        self.loss_cycle_A = self.loss_cycle_A.data.item()
        self.loss_cycle_B = self.loss_cycle_B.data.item()
        self.loss_ident_A = self.loss_ident_A.data.item()
        self.loss_ident_B = self.loss_ident_B.data.item()

    def backward_D(self):

        loss_dis_A, self.losses_adv_A, _ = self.crit_dis_A(
            img_real_dst=self.real_A, 
            img_fake_dst=self.fake_A.detach(),
            enc=self.enc)

        loss_dis_B, self.losses_adv_B, _ = self.crit_dis_B(
            img_real_dst=self.real_B, 
            img_fake_dst=self.fake_B.detach(),
            enc=self.enc)

        loss_D = loss_dis_A + loss_dis_B

        if self.training:
            loss_D.backward()

    def train(self, mode=True):

        self.training = mode
        
        self.gen_A.train(mode)
        self.gen_B.train(mode)
        self.dis_A.train(mode)
        self.dis_B.train(mode)
        self.crit_dis_A.train(mode)
        self.crit_dis_B.train(mode)

        return self
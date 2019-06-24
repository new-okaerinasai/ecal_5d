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
        self.gen_B = Generator(opt, 'B', opt.gen_type_name_B)

        # Discriminators
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

        print('\nDis B\n')
        num_params = 0
        for p in self.dis_B.parameters():
            num_params += p.numel()
        print(self.dis_B)
        print('Number of parameters: %d' % num_params)

        self.gen_params = self.gen_B.parameters()

        self.dis_params = self.dis_B.parameters()

        # Losses
        self.crit_dis_B = DiscriminatorLoss(opt, self.dis_B)

        # If an encoder is required, load the weights
        if (opt.mse_loss_type_B == 'perceptual' or
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

        # Pretrained aux classifier/regressor
        if opt.pretrained_aux_path:

            self.aux = torch.load(opt.pretrained_aux_path)

            self.crit_aux_B = utils.get_criterion(
                opt.aux_loss_type, 
                opt.gen_aux_loss_weight,
                self.enc)

            print('')
            print(self.aux)
            print('')

        # In case domains have different sizes, this is needed for mse loss
        scale_factor = opt.img_size_B // opt.img_size_A

        self.down = nn.AvgPool2d(scale_factor)
        self.up = nn.Upsample(
            scale_factor=scale_factor, 
            mode='bilinear',
            align_corners=False)

        # Load onto gpus
        self.gen_B = nn.DataParallel(self.gen_B.cuda(self.gpu_id), opt.gpu_ids)
        self.dis_B = nn.DataParallel(self.dis_B.cuda(self.gpu_id), opt.gpu_ids)
        if hasattr(self, 'aux'):
            self.aux = nn.DataParallel(self.aux.cuda(self.gpu_id), opt.gpu_ids)
        if self.enc is not None: 
            self.enc = nn.DataParallel(self.enc.cuda(self.gpu_id), opt.gpu_ids)

    def forward(self, inputs):

        if len(inputs) == 4:
            real_A, real_A_aux, _, real_B_aux = inputs

        # Input images
        self.real_A = Variable(real_A.cuda(self.gpu_id))

        self.real_A_aux = Variable(real_A_aux.cuda(self.gpu_id))
        self.real_B_aux = Variable(real_B_aux.cuda(self.gpu_id))

        # Fake images      
        self.fake_B = self.gen_B(self.real_A, self.real_B_aux)

    def backward_G(self):

        # Cycle loss
        cycle_A = self.gen_B(self.down(self.fake_B), self.real_A_aux)

        self.loss_cycle_A = self.crit_mse_A(cycle_A, self.real_A)

        # MSE loss
        loss_mse_A = self.loss_cycle_A

        # GAN loss
        loss_dis_B, _, _ = self.crit_dis_B(
            img_real_dst=self.fake_B,
            aux_real_dst=self.real_B_aux,
            enc=self.enc)

        loss_G = loss_mse_A + loss_dis_B

        if hasattr(self, 'crit_aux_B'):
            fake_B_aux = self.aux(self.fake_B)
            self.loss_auxil_B = self.crit_aux_B(fake_B_aux, self.real_B_aux)
            loss_G += self.loss_auxil_B

        if self.training:
            loss_G.backward()

        # Get values for visualization
        self.loss_cycle_A = self.loss_cycle_A.data.item()

        # Get values for visualization
        if hasattr(self, 'crit_aux_B'):
            self.loss_auxil_B = self.loss_auxil_B.data.item()

    def backward_D(self):

        loss_dis_B, self.losses_adv_B, losses_aux_B = self.crit_dis_B(
            img_real_dst=self.real_A, 
            img_fake_dst=self.fake_B.detach(),
            aux_real_dst=self.real_A_aux,
            enc=self.enc)

        if losses_aux_B: self.losses_aux_B = losses_aux_B

        loss_D = loss_dis_B

        if self.training:
            loss_D.backward()

    def train(self, mode=True):

        self.training = mode
        
        self.gen_B.train(mode)
        self.dis_B.train(mode)
        self.crit_dis_B.train(mode)

        return self
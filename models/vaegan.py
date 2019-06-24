import torch
from torch import nn
from .variational_generator import Generator
from .discriminator_wrapper import DiscriminatorWrapper
from .discriminator_loss import DiscriminatorLoss
from .perceptual_loss import FeatureExtractor
from src import utils
from torch.autograd import Variable
from torchvision.models import resnet18, vgg19
import os



class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()

        self.gpu_id = opt.gpu_ids[0]
        self.weights_path = os.path.join(opt.experiment_path, 'checkpoints')

        # Generator
        self.gen_B = Generator(opt, 'B', opt.gen_type_name_B)
        
        self.noise_size = (opt.batch_size, self.gen_B.noise_channels)

        # Discriminator
        if opt.dis_type_names_B[0]: self.dis_B = DiscriminatorWrapper(opt, 'B')

        # Load weights
        utils.load_checkpoint(self, opt.which_epoch, opt.pretrained_gen_path)

        # Print architectures
        print('\nGen A to B\n')
        num_params = 0
        for p in self.gen_B.parameters():
            num_params += p.numel()
        print(self.gen_B)
        print('Number of parameters: %d' % num_params)

        self.gen_params = self.gen_B.parameters()

        # Discriminator
        if opt.dis_type_names_B[0]:

            print('\nDis B\n')
            num_params = 0
            for p in self.dis_B.parameters():
                num_params += p.numel()
            print(self.dis_B)
            print('Number of parameters: %d' % num_params)

            self.dis_params = self.dis_B.parameters()

            # Losses
            self.crit_dis_B = DiscriminatorLoss(opt, self.dis_B)

        # If an encoder is required, load the weights
        if (opt.mse_loss_type == 'perceptual' or 
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

        self.crit_mse = utils.get_criterion(
            opt.mse_loss_type, 
            opt.mse_loss_weight,
            self.enc)

        # In case domains have different sizes, this is needed for mse loss
        scale_factor = opt.img_size_B // opt.img_size_A

        self.up = nn.Upsample(
            scale_factor=1, 
            mode='bilinear',
            align_corners=False)

        # Load onto gpus
        self.gen_B = nn.DataParallel(self.gen_B.cuda(self.gpu_id), opt.gpu_ids)
        if opt.dis_type_names_B[0]:
            self.dis_B = nn.DataParallel(self.dis_B.cuda(self.gpu_id), opt.gpu_ids)
        if self.enc is not None: 
            self.enc = nn.DataParallel(self.enc.cuda(self.gpu_id), opt.gpu_ids)

    def forward(self, inputs):

        if len(inputs) == 1:
            real_B = inputs[0]
        elif len(inputs) == 2:
            real_B, real_B_aux = inputs

        # Input images
        self.real_B = Variable(real_B.cuda(self.gpu_id))

        noise = Variable(torch.randn(self.noise_size).cuda(self.gpu_id))

        # Reconstruction images
        self.ident_B, self.mu, self.logvar = self.gen_B(noise, self.real_B)

        # Fake images
        if hasattr(self, 'crit_dis_B'):
            self.fake_B = self.gen_B(noise)
        else:
            self.fake_B = self.ident_B

    def backward_G(self):

        # Identity loss
        self.loss_ident_B = self.crit_mse(self.ident_B, self.real_B)

        loss_mse_B = self.loss_ident_B

        # KL divergence between q(z|x) and p(z)
        self.loss_kldiv_B = -0.5 * (
            1 + self.logvar - self.mu.pow(2) - self.logvar.exp()).sum(1).mean(0)

        loss_vae_B = self.loss_kldiv_B

        # GAN loss
        if hasattr(self, 'crit_dis_B'):
            loss_dis_B, _, _ = self.crit_dis_B(
                img_real_dst=self.fake_B,
                enc=self.enc)
        else:
            loss_dis_B = 0

        loss_G = loss_mse_B + loss_vae_B + loss_dis_B
        
        if self.training:
            loss_G.backward()

        # Get values for visualization
        self.loss_ident_B = self.loss_ident_B.data.item()
        self.loss_kldiv_B = self.loss_kldiv_B.data.item()

    def backward_D(self):

        loss_dis_B, self.losses_adv_B, _ = self.crit_dis_B(
            img_real_dst=self.real_B, 
            img_fake_dst=self.fake_B.detach(),
            enc=self.enc)

        loss_D = loss_dis_B

        if self.training:
            loss_D.backward()

    def train(self, mode=True):

        self.training = mode
        
        self.gen_B.train(mode)
        if hasattr(self, 'dis_B'):
            self.dis_B.train(mode)
            self.crit_dis_B.train(mode)

        return self
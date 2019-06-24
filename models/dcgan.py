import torch
from torch import nn
from .generator import Generator
from .discriminator_wrapper import DiscriminatorWrapper
from .discriminator_loss import DiscriminatorLoss
from .perceptual_loss import FeatureExtractor
from src import utils
from torch.autograd import Variable
from torchvision.models import resnet18, vgg19
import os
import torch.nn.functional as F
import numpy as np
#import nirvana_dl
input_path = "./"#nirvana_dl.get_input_path()

class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()

        self.gpu_id = opt.gpu_ids[0]
        self.weights_path = os.path.join(opt.experiment_path, 'checkpoints')

        # Generator
        self.gen_B = Generator(opt, 'B', opt.gen_type_name_B)
        
        self.noise_size = (opt.batch_size, self.gen_B.noise_channels)

        # Discriminator
        if opt.dis_type_names_B: self.dis_B = DiscriminatorWrapper(opt, 'B')

        # Load weights
        utils.load_checkpoint(self, opt.which_epoch, opt.pretrained_gen_path)

        # Print architectures
        print('\nGen A to B\n')
        num_params = 0
        for p in self.gen_B.parameters():
            num_params += p.numel()
        print(self.gen_B)
        print('Number of parameters: %d' % num_params)

        self.X_min = torch.from_numpy(np.load(os.path.join(input_path, "data_min.npy")))
        self.X_min = self.X_min.cuda()

        self.X_max = torch.from_numpy(np.load(os.path.join(input_path, "data_max.npy")))
        self.X_max = self.X_max.cuda()
        
        self.X_mean = torch.from_numpy(np.load(os.path.join(input_path, "data_mean.npy")))
        self.X_mean = self.X_mean.cuda()

        self.X_std = torch.from_numpy(np.load(os.path.join(input_path, "data_std.npy")))
        self.X_std = self.X_std.cuda()
        
        self.y_std = torch.from_numpy(np.load(os.path.join(input_path, "target_mean.npy")))
        self.y_std = self.y_std.cuda()

        self.y_mean = torch.from_numpy(np.load(os.path.join(input_path, "target_std.npy")))
        self.y_mean = self.y_mean.cuda()

        self.gen_params = self.gen_B.parameters()

        # Discriminator
        if opt.dis_type_names_B:

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
        if hasattr(self, 'dis_B') and self.dis_B.use_encoder:

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

        self.up = nn.Upsample(
            scale_factor=1, 
            mode='bilinear',
            align_corners=False)

        # Load onto gpus
        self.gen_B = nn.DataParallel(self.gen_B.cuda(self.gpu_id), opt.gpu_ids)
        if opt.dis_type_names_B:
        	self.dis_B = nn.DataParallel(self.dis_B.cuda(self.gpu_id), opt.gpu_ids)
        if hasattr(self, 'aux'):
            self.aux = nn.DataParallel(self.aux.cuda(self.gpu_id), opt.gpu_ids)
        if self.enc is not None: 
            self.enc = nn.DataParallel(self.enc.cuda(self.gpu_id), opt.gpu_ids)
    
    def denormalize_data(self, fake_imgs):
        X = fake_imgs[:, 0, ...].double()
        X = X / 2 + 0.5
        X = X * (self.X_max - self.X_min) + self.X_min
        X = X*self.X_std + self.X_mean
        X = torch.exp(X)
        return X

    def denormalize_target(self, y):
        o = y.double() * self.y_std + self.y_mean
        return torch.exp(o)

    def forward(self, inputs):

        if len(inputs) == 1:
            real_B = inputs[0]
        elif len(inputs) == 2:
            real_B, real_B_aux = inputs

        # Input images
        self.real_B = Variable(real_B.cuda(self.gpu_id))
        self.real_B_denormalized = self.denormalize_data(self.real_B)
        #if 'real_B_aux' in locals():
        self.real_B_aux = Variable(real_B_aux.cuda(self.gpu_id))
        self.real_B_aux_denormalized = self.denormalize_target(self.real_B_aux)
        #else:
        #    self.real_B_aux = None

        noise = Variable(torch.randn(self.noise_size).cuda(self.gpu_id))

        self.gen_B.cuda()

        # Fake images
        self.fake_B = self.gen_B(noise, self.real_B_aux)
        self.fake_B_denormalized = self.denormalize_data(self.fake_B)

    def get_assymetry(self, data, ps, points, orthog=False):
        # асимметрия ливня вдоль и поперек направнения наклона
        first = True
        assym_res = []
        for i in range(len(data)):
            img = data[i]
            p = ps[i]
            #print('momentum', p)
            point = points[i, :2]
    #        zoff = 50
            zoff = 25
            point0 = point[0] + zoff*p[0]/p[2]
            point1 = point[1] + zoff*p[1]/p[2]
            if orthog:
                line_func = lambda x: (x.float() - point0.float()) / p[0].float() * p[1].float() + point1.float()
            else:
                line_func = lambda x: -(x.float() - point0.float()) / p[1].float() * p[0].float() + point1.float()



            x = torch.linspace(-14.5, 14.5, 32)
            y = torch.linspace(-14.5, 14.5, 32)
            xx, yy = torch.meshgrid([x, y])
            xx = torch.transpose(xx, 0, 1).cuda()
            yy = torch.transpose(yy, 0, 1).cuda()
            zz = torch.where(yy - line_func(xx) < 0, torch.zeros_like(yy).cuda(), torch.ones_like(yy).cuda())
            if (not orthog and p[1]<0):
                zz = torch.where(yy - line_func(xx) > 0, torch.zeros_like(yy).cuda(), torch.ones_like(yy).cuda())
            
            assym = (torch.sum(img.float() * zz.float()) - torch.sum(img.float() * (1 - zz))) / torch.sum(img.float())
            assym_res.append(assym[None])

        return torch.cat(assym_res)

    def loss_quantile(self, fake_data, real_data, fake_aux, real_aux, orthog=False):
        assym1 = self.get_assymetry(real_data, real_aux[:, :3], real_aux[:, 3:], orthog)
        assym2 = self.get_assymetry(fake_data, fake_aux[:, :3], fake_aux[:, 3:], orthog)

        assym1, _ = torch.sort(assym1)
        assym2, _ = torch.sort(assym1)

        return F.mse_loss(assym1, assym2)
        
    def backward_G(self):

        loss_G = 0

        # GAN loss
        if hasattr(self, 'crit_dis_B'):
            loss_dis_B, _, losses_aux_B = self.crit_dis_B(
                img_real_dst=self.fake_B,
                aux_real_dst=self.real_B_aux,
                enc=self.enc)
            loss_G += loss_dis_B
        
        if hasattr(self, 'crit_aux_B'):
            fake_B_aux = self.aux(self.fake_B)
            self.loss_auxil_B = self.crit_aux_B(fake_B_aux, self.real_B_aux)
            loss_G += self.loss_auxil_B
            loss_G += self.loss_quantile(self.fake_B_denormalized,
                                         self.real_B_denormalized,
                                         self.denormalize_target(fake_B_aux), 
                                         self.real_B_aux_denormalized,
                                         False)

            loss_G += self.loss_quantile(self.fake_B_denormalized,
                                         self.real_B_denormalized,
                                         self.denormalize_target(fake_B_aux),
                                         self.real_B_aux_denormalized,
                                         True)
        if self.training and loss_G:
            loss_G.backward()
        # Get values for visualization
        if hasattr(self, 'crit_aux_B'):
            self.loss_auxil_B = self.loss_auxil_B.data.item()

    def backward_D(self):

        loss_dis_B, self.losses_adv_B, losses_aux_B = self.crit_dis_B(
            img_real_dst=self.real_B, 
            img_fake_dst=self.fake_B.detach(),
            aux_real_dst=self.real_B_aux,
            enc=self.enc)

        if losses_aux_B: self.losses_aux_B = losses_aux_B

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

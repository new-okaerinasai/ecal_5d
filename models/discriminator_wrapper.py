import torch
from torch import nn
from src import utils
from .discriminator import Discriminator



class DiscriminatorWrapper(nn.Module):

    def __init__(self, opt, domain):
        super(DiscriminatorWrapper, self).__init__()

        opt = vars(opt)

        self.domain = domain
        self.down_fracs = opt['dis_down_fracs_%s' % domain]
        type_names = opt['dis_type_names_%s' % domain]

        self.downs = {}
        self.discs = nn.ModuleList()

        self.adv_loss_types = []
        self.shapes = []
        self.weights = []

        self.enc_fracs = []
        self.enc_idx = []

        for frac, name in zip(self.down_fracs, type_names):

            down = nn.AvgPool2d(frac)
            disc = Discriminator(opt, name)

            self.adv_loss_types += [disc.adv_loss_type]
            self.shapes += [disc.output_shapes]
            self.weights += [disc.output_weights]

            if frac not in self.downs.keys():
                self.downs[frac] = down
                if disc.use_encoder:
                    self.enc_fracs += [frac]

            if disc.use_encoder: 
                self.enc_idx += [len(self.discs)]
            
            self.discs += [disc]

        self.use_encoder = len(self.enc_idx) > 0

        # In case we're solving super resolution problem
        self.scale_factor = opt['img_size_B'] // opt['img_size_A']

        self.up = nn.Upsample(
            scale_factor=self.scale_factor, 
            mode='bilinear',
            align_corners=False)

    def forward(self, img_dst, img_src=None, enc=None, dis_idx=None):

        if dis_idx is None: dis_idx = list(range(len(self.discs)))

        # Dictionary of downsampling fraction: images/features at that scale
        imgs = {}
        feats = {}

        down_fracs = [self.down_fracs[i] for i in dis_idx]

        for frac, down in zip(self.downs.keys(), self.downs.values()):
            
            if frac not in down_fracs:
                continue

            # Downsample imgs from domain B
            imgs[frac] = down(img_dst)

            if frac in self.enc_fracs and enc is not None:
                
                # Downsample imgs from domain A and concatenate them
                if img_src is not None and self.scale_factor == 1:
                    imgs[frac] = torch.cat([
                        imgs[frac], 
                        down(self.up(img_src))], 0)

                # Get feats for both imgs A and imgs B
                feats[frac] = enc(imgs[frac])

                if img_src is not None and self.scale_factor == 1:

                    # Reshape feats from 2BxCxHxW to Bx2CxHxW
                    feats_reshaped = []
                    for f in feats[frac]:
                        b, c, h, w = f.shape
                        feats_reshaped += [f.view(b//2, c*2, h, w)]
                    feats[frac] = feats_reshaped

                    # Same for imgs
                    b, c, h, w = imgs[frac].shape
                    imgs[frac] = imgs[frac].view(b//2, c*2, h, w)
            else:

                if img_src is not None and self.scale_factor == 1:
                    imgs[frac] = torch.cat([
                        imgs[frac], 
                        down(self.up(img_src))], 1)

        outputs = []
        outputs_aux = []

        discs = [self.discs[i] for i in dis_idx]

        for i, frac, disc in zip(dis_idx, down_fracs, discs):

            # Get predictions from discriminator
            if i in self.enc_idx:
                output, output_aux = disc(feats[frac])
            else:
                output, output_aux = disc([imgs[frac]])

            outputs += [output]
            outputs_aux += [output_aux]

        return outputs, outputs_aux

    def train(self, mode=True):

        self.training = mode
        
        self.discs.train(mode)

        return self

    def __len__(self):

        return len(self.discs)

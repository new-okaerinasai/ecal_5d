import torch
from torch import nn
from torch.autograd import Variable
from math import log
from src import utils



def get_args(parser, type_name):
    """Add options to define a generator of specified type.

    Args:
        type_name: Specified generator type name.
    """

    parser.add('--%s_noise_channels' % type_name, 
               default=100, type=int,
               help='amount of noise channels to sample from')

    parser.add('--%s_num_channels' % type_name, 
               default=0, type=int,
               help='overwrites global num_channels for this network if != -1')

    parser.add('--%s_max_channels' % type_name, 
               default=0, type=int,
               help='overwrites global max_channels for this network if != -1')

    parser.add('--%s_norm_layer' % type_name, 
               default='batch', type=str,
               help='name of normalization layer, instance|batch|l2|l1|none')

    parser.add('--%s_upsampling_layer' % type_name, 
               default='conv_transpose', type=str,
               help='upsampling module, conv_transpose|nearest|bilinear')

    parser.add('--%s_kernel_size' % type_name, 
               default=4, type=int,
               help='kernel size for downsampling and upsampling convolutions')

    parser.add('--%s_nonlinear_layer' % type_name, 
               default='relu', type=str,
               help='type of nonlinearity, relu|leakyrelu|swish|tanh')

    parser.add('--%s_output_range' % type_name, 
               default='tanh', type=str,
               help='range of output tensor, relu|tanh|threshold')


class Generator(nn.Module):

    def __init__(self, opt, domain, type_name):
        super(Generator, self).__init__()

        opt = vars(opt)

        # Check which mapping does this generator perform
        ZtoB = domain == 'B'

        # Read options
        img_channels = opt['img_channels_%s' % ('A' if ZtoB else 'B')]
        output_size = opt['img_size_%s' % 'B' if ZtoB else 'A']
        output_channels = opt['img_channels_%s' % 'B' if ZtoB else 'A']
        self.noise_channels = opt['%s_noise_channels' % type_name]
        num_channels = opt['%s_num_channels' % type_name]
        max_channels = opt['%s_max_channels' % type_name]
        kernel_size = opt['%s_kernel_size' % type_name]
        norm_layer = utils.get_norm_layer(opt['%s_norm_layer' % type_name])
        upsampling_layer = opt['%s_upsampling_layer' % type_name]
        nonlinear_layer = utils.get_nonlinear_layer(
            opt['%s_nonlinear_layer' % type_name])
        final_nonlinear_layer = utils.get_nonlinear_layer(
            opt['%s_output_range' % type_name])
        aux_channels = opt['aux_channels']

        # Setup number of channels
        if not num_channels: num_channels = opt['num_channels']
        if not max_channels: max_channels = opt['max_channels']

        # Calculate network depth
        depth = int(log(output_size // 4, 2))

        # Encoder
        in_channels = img_channels
        out_channels = num_channels

        layers = [
            nn.Conv2d(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=7, 
                stride=1, 
                padding=3, 
                bias=False),
            nonlinear_layer(True)]

        for i in range(depth):

            in_channels = out_channels
            out_channels = min(num_channels * 2**(i+1), max_channels)

            layers += utils.get_conv_block(
                in_channels=in_channels, 
                out_channels=out_channels,
                nonlinear_layer=nonlinear_layer, 
                norm_layer=norm_layer,
                mode='down', 
                sequential=False,
                kernel_size=kernel_size)

        in_channels = out_channels

        layers += [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.noise_channels*2,
                kernel_size=4,
                stride=4),
            utils.View()]

        self.encoder = nn.Sequential(*layers)

        # Decoder
        in_channels = self.noise_channels
        out_channels = min(num_channels * 2**depth, max_channels)
        
        layers = [utils.View()]

        layers += utils.get_upsampling_block(
            in_channels=in_channels,
            out_channels=out_channels,
            nonlinear_layer=nonlinear_layer,
            norm_layer=norm_layer,
            mode=upsampling_layer,
            sequential=False,
            kernel_size=kernel_size,
            factor=4)

        for i in range(depth):

            in_channels = out_channels
            out_channels = min(num_channels * 2**(depth-1-i), max_channels)

            layers += utils.get_upsampling_block(
                in_channels=in_channels,
                out_channels=out_channels,
                nonlinear_layer=nonlinear_layer,
                norm_layer=norm_layer,
                mode=upsampling_layer,
                sequential=False,
                kernel_size=kernel_size,)

        layers += [
            nn.Conv2d(
                in_channels=out_channels, 
                out_channels=output_channels, 
                kernel_size=7, 
                stride=1, 
                padding=3, 
                bias=False),
            final_nonlinear_layer(True)]

        self.decoder = nn.Sequential(*layers)

        # Initialize weights
        self.apply(utils.weights_init)

    def reparameterize(self, mu, logvar, eps):

        if self.training:
        
            std = torch.exp(0.5*logvar)

            return eps.mul(std).add_(mu)
        
        else:

            return mu

    def forward(self, eps, img=None):

        if img is None:

            return self.decoder(eps)

        else:

            mu, logvar = torch.split(self.encoder(img), self.noise_channels, 1)

            z = self.reparameterize(mu, logvar, eps)

            return self.decoder(z), mu, logvar

import torch
from torch import nn
from src import utils



def get_args(parser, type_name):
    """Add options to define a generator of specified type.

    For each domain, separate generator can be specified (useful when domains 
    are not symmetric, ex. superresolution).

    Args:
        type_name: Specified generator type name.
    """

    parser.add('--%s_num_channels' % type_name, 
               default=0, type=int,
               help='overwrites global num_channels for this network if != 0')

    parser.add('--%s_max_channels' % type_name, 
               default=0, type=int,
               help='overwrites global max_channels for this network if != 0')

    parser.add('--%s_num_down_blocks' % type_name, 
               default=3, type=int,
               help='number of downsampling blocks')

    parser.add('--%s_num_res_blocks' % type_name, 
               default=8, type=int,
               help='number of residual blocks operating on latent space')

    parser.add('--%s_res_block_type' % type_name, 
               default='conv', type=str,
               help='type of the residual block used, conv|attention')

    parser.add('--%s_num_up_blocks' % type_name, 
               default=3, type=int,
               help='number of upsampling blocks')

    parser.add('--%s_norm_layer' % type_name, 
               default='none', type=str,
               help='name of normalization layer, instance|batch|l2|l1|none')

    parser.add('--%s_upsampling_layer' % type_name, 
               default='conv_transpose', type=str,
               help='upsampling module, conv_transpose|nearest|bilinear')

    parser.add('--%s_kernel_size' % type_name, 
               default=4, type=int,
               help='kernel size for downsampling and upsampling convolutions')

    parser.add('--%s_nonlinear_layer' % type_name, 
               default='relu', type=str,
               help='type of nonlinearity, relu|leakyrelu|swish|tanh|none')

    parser.add('--%s_output_range' % type_name, 
               default='tanh', type=str,
               help='range of output tensor, relu|tanh|none')


class Generator(nn.Module):
    """Convolutional generator."""

    def __init__(self, opt, domain, type_name):
        super(Generator, self).__init__()

        opt = vars(opt)

        # Check which mapping does this generator perform
        AtoB = domain == 'B'

        # Read options
        img_channels = opt['img_channels_%s' % ('A' if AtoB else 'B')]
        aux_channels = opt['aux_channels_%s' % ('A' if AtoB else 'B')]
        output_channels = opt['img_channels_%s' % ('B' if AtoB else 'A')]
        num_channels = opt['%s_num_channels' % type_name]
        max_channels = opt['%s_max_channels' % type_name]
        num_down_blocks = opt['%s_num_down_blocks' % type_name]
        num_res_blocks = opt['%s_num_res_blocks' % type_name]
        res_block_type = opt['%s_res_block_type' % type_name]
        num_up_blocks = opt['%s_num_up_blocks' % type_name]
        kernel_size = opt['%s_kernel_size' % type_name]
        norm_layer = utils.get_norm_layer(opt['%s_norm_layer' % type_name])
        upsampling_layer = opt['%s_upsampling_layer' % type_name]
        nonlinear_layer = utils.get_nonlinear_layer(
            opt['%s_nonlinear_layer' % type_name])
        final_nonlinear_layer = utils.get_nonlinear_layer(
            opt['%s_output_range' % type_name])

        # Setup number of channels
        if not num_channels: num_channels = opt['num_channels']
        if not max_channels: max_channels = opt['max_channels']

        # Setup res block
        if res_block_type == 'conv':
            res_block = utils.ResBlock
        elif res_block_type == 'attention':
            res_block = utils.LocalAttention

        # Calculate representation depth
        depth = max(num_down_blocks, num_up_blocks)

        # First layer has no normalization
        in_channels = img_channels + aux_channels
        out_channels = min(
            num_channels * 2**(depth-num_down_blocks), 
            max_channels)

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels, 
                7, 1, 3, 
                bias=False),
            nonlinear_layer(True)]

        # Downsampling blocks
        for i in range(num_down_blocks):

            in_channels = out_channels
            out_channels = min(
                num_channels * 2**(depth-num_down_blocks+i+1), 
                max_channels)

            layers += utils.get_conv_block(
                in_channels=in_channels, 
                out_channels=out_channels,
                nonlinear_layer=nonlinear_layer, 
                norm_layer=norm_layer,
                mode='down', 
                sequential=False,
                kernel_size=kernel_size)

        in_channels = out_channels

        # Residual blocks
        for i in range(num_res_blocks):

            layers += [res_block(
                in_channels=in_channels, 
                nonlinear_layer=nonlinear_layer,
                norm_layer=norm_layer)]

        # Upsampling blocks
        for i in range(num_up_blocks):

            in_channels = out_channels
            out_channels = min(
                num_channels * 2**(depth-i-1), 
                max_channels)

            layers += utils.get_upsampling_block(
                in_channels=in_channels,
                out_channels=out_channels,
                nonlinear_layer=nonlinear_layer,
                norm_layer=norm_layer,
                mode=upsampling_layer,
                sequential=False,
                kernel_size=kernel_size)

        # Final block
        layers += [
            nn.Conv2d(
                out_channels,
                output_channels, 
                7, 1, 3, 
                bias=False),
            final_nonlinear_layer(True)]

        self.generator = nn.Sequential(*layers)

        # Initialize weights
        self.apply(utils.weights_init)

    def forward(self, img, aux=None):

        if aux is not None:

            b, c = aux.shape
            _, _, h, w = img.shape

            aux = aux.view(b, c, 1, 1).repeat(1, 1, h, w)
            
            input = torch.cat([img, aux], 1)

        else:

            input = img

        return self.generator(input)

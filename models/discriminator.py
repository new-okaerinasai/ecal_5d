import torch
from torch import nn
from math import log
from src import utils



def get_args(parser, type_name):
    """Add options to define a discriminator of specified type.

    There is a list of discriminator type names, "dis_type_names", which 
    allows the definition of multiple discriminators with the same 
    hyperparameters. Later are specified in options with "type_name" prefix. 
    This function adds these options into a parser for each input type_name.

    Args:
        type_name: Specified discriminator type name.
    """

    parser.add('--%s_input_sizes' % type_name,
               default='256,128,64,32,16', type=str,
               help='list of input tensors spatial sizes separated by commas')

    parser.add('--%s_output_sizes' % type_name,
               default='8,4,1', type=str,
               help='list of output tensors spatial sizes separated by commas')

    parser.add('--%s_output_weights' % type_name,
               default='1,1,1', type=str,
               help='discriminator output weights')

    parser.add('--%s_input_num_channels' % type_name, 
               default='64,128,256,512,512', type=str,
               help='list of num channels in the input tensors')

    parser.add('--%s_num_channels' % type_name, 
               default=0, type=int,
               help='overwrites global num_channels for this network if != 0')

    parser.add('--%s_max_channels' % type_name, 
               default=0, type=int,
               help='overwrites global max_channels for this network if != 0')

    parser.add('--%s_adv_loss_type' % type_name, 
               default='gan', type=str,
               help='loss type for probability preds, gan|lsgan|wgan')

    parser.add('--%s_use_encoder' % type_name, 
               action='store_true',
               help='use pretrained encoder outputs as inputs')

    parser.add('--%s_norm_layer' % type_name, 
               default='none', type=str,
               help='type of normalization layer, instance|batch|l2|l1|none')

    parser.add('--%s_norm_layer_cat' % type_name, 
               default='l2', type=str,
               help='type of norm after concat, instance|batch|l2|l1|none')

    parser.add('--%s_kernel_size' % type_name, 
               default=4, type=int,
               help='kernel size for downsampling convolutions')

    parser.add('--%s_kernel_size_io' % type_name, 
               default=3, type=int,
               help='kernel size for input and output convolutions')

    parser.add('--%s_nonlinear_layer' % type_name, 
               default='leakyrelu', type=str,
               help='type of nonlinearity, relu|leakyrelu|swish|tanh|sigmoid|none')


class Discriminator(nn.Module):
    """Convolutional discriminator.

    This is a convolutional discriminator network. It receives and outputs 
    tensors of specified spatial sizes. For more detailed descriptions of 
    options see help of get_args function above.

    Attributes:
        input_sizes: List of spatial sizes for inputs.
        adv_loss_type: Loss type for probability preds.
        use_encoder: Use pretrained encoder outputs as inputs.
        depth: Amount of downsampling convolutional blocks.
        output_shape: Shape of the network output.
        blocks: Downsampling convolutional blocks.
        output_blocks: Convolutional blocks used to output probabilities.
        concat_blocks: Convolutional blocks used to input tensors (optional).
        concat_blocks_depth: Depth at which each input tensor is concatenated.
        output_blocks_depth: Depth at which each prediction is output.
    """

    def __init__(self, opt, type_name):
        """Initialize discriminator network.

        This method creates and initializes trainable discriminator layers.

        Args:
            opt: Options specified by the user.
            type_name: Specified discriminator type name.
        """
        super(Discriminator, self).__init__()

        # Read options
        self.input_sizes = opt['%s_input_sizes' % type_name].split(',')
        self.input_sizes = [int(i) for i in self.input_sizes]
        output_sizes = opt['%s_output_sizes' % type_name].split(',')
        self.output_sizes = [int(i) for i in output_sizes]
        output_weights = opt['%s_output_weights' % type_name].split(',')
        self.output_weights = [float(i) for i in output_weights]
        kernel_size = opt['%s_kernel_size' % type_name]
        kernel_size_io = opt['%s_kernel_size_io' % type_name]
        input_num_channels = opt['%s_input_num_channels' % type_name].split(',')
        input_num_channels = [int(i) for i in input_num_channels]
        num_channels = opt['%s_num_channels' % type_name]
        max_channels = opt['%s_max_channels' % type_name]
        self.adv_loss_type = opt['%s_adv_loss_type' % type_name]
        self.use_encoder = opt['%s_use_encoder' % type_name]
        norm_layer = utils.get_norm_layer(opt['%s_norm_layer' % type_name])
        norm_layer_cat = utils.get_norm_layer(
            opt['%s_norm_layer_cat' % type_name])
        nonlinear_layer = utils.get_nonlinear_layer(
            opt['%s_nonlinear_layer' % type_name])
        aux_channels = opt['aux_channels']

        # Setup number of channels
        if not num_channels: num_channels = opt['num_channels']
        if not max_channels: max_channels = opt['max_channels']

        # Calculate final tensor spatial size
        if (self.output_sizes[-1] == 2  or 
            len(self.output_sizes) > 1 and self.output_sizes[-2] < 4):
            result_spatial_size = self.output_sizes[-2]
        else:
            result_spatial_size = 4

        # Calculate network depth
        self.depth = int(log(self.input_sizes[0] // result_spatial_size, 2))

        # Calculate output shape
        self.output_shapes = []
        for s in self.output_sizes:
            self.output_shapes += [torch.Size([opt['batch_size'], s**2])]

        # Initialize network
        self.blocks = nn.ModuleList()
        self.concat_blocks = nn.ModuleList()
        self.output_blocks = nn.ModuleList()

        in_channels = input_num_channels[0]
        out_channels = num_channels
        current_size = self.input_sizes[0]

        self.concat_blocks_depth = []
        self.output_blocks_depth = []

        for i in range(self.depth):

            # Define downsampling block
            self.blocks += [utils.get_conv_block(
                in_channels=in_channels, 
                out_channels=out_channels,
                nonlinear_layer=nonlinear_layer, 
                norm_layer=norm_layer if i else utils.get_norm_layer('none'),
                mode='down', 
                sequential=True,
                kernel_size=kernel_size)]

            current_size //= 2

            in_channels = out_channels

            if current_size in self.input_sizes:

                k = self.input_sizes.index(current_size)

                # Define concat block
                self.concat_blocks += [utils.ConcatBlock(
                    enc_channels=input_num_channels[k],
                    out_channels=in_channels,
                    nonlinear_layer=nonlinear_layer,
                    norm_layer=norm_layer,
                    norm_layer_cat=norm_layer_cat,
                    kernel_size=kernel_size_io)]

                self.concat_blocks_depth += [i]

                in_channels *= 2

            if current_size in self.output_sizes:

                # Define PatchGAN output block
                self.output_blocks += [nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels, 
                        out_channels=1, 
                        kernel_size=kernel_size_io, 
                        stride=1, 
                        padding=kernel_size_io//2, 
                        bias=False),
                    utils.View())]

                self.output_blocks_depth += [i]

            out_channels = min(out_channels * 2, max_channels)

        if 1 in self.output_sizes:

            # Final probability prediction
            self.output_blocks += [nn.Sequential(
                nn.Conv2d(
                    in_channels=out_channels, 
                    out_channels=1, 
                    kernel_size=current_size, 
                    stride=current_size, 
                    padding=0, 
                    bias=False),
                utils.View())]

        # Auxiliary prediction
        #if aux_channels and opt['dis_aux_loss_weight']:

        #    self.aux_output_block = nn.Sequential(nn.Conv2d(
        #        in_channels=out_channels, 
        #        out_channels=aux_channels, 
        #        kernel_size=current_size, 
        #        stride=current_size,
        #        padding=0,
        #        bias=False),
        #        utils.View())

        # Initialize weights
        self.apply(utils.weights_init)

    def forward(self, inputs):

        result = inputs[0]

        # Specify current indices for concat and output blocks
        cur_concat_idx = 0
        cur_output_idx = 0

        # Probability output
        output = []

        for i in range(self.depth):

            result = self.blocks[i](result)
            
            if i in self.concat_blocks_depth:
                
                # Concatenate next input to current result
                result = self.concat_blocks[cur_concat_idx](
                    result, 
                    inputs[cur_concat_idx+1])
                
                cur_concat_idx += 1

            if i in self.output_blocks_depth:
                
                # Output probabilities in PatchGAN style
                output += [self.output_blocks[cur_output_idx](result)]
                
                cur_output_idx += 1

        if 1 in self.output_sizes:
            
            # Final probability prediction
            output += [self.output_blocks[cur_output_idx](result)]

        if hasattr(self, 'aux_output_block'):
            
            # Get auxiliary prediction
            output_aux = self.aux_output_block(result)

        else:

            output_aux = None

        return output, output_aux

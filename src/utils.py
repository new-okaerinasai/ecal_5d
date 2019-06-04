import torch
from torch import nn
import torch.nn.functional as f
from models.perceptual_loss import PerceptualLoss
import os



def get_criterion(loss, weight=1., encoder=None):

    loss = loss.lower()

    if loss == 'l1':
        loss_func = nn.L1Loss()
    elif loss == 'l2':
        loss_func = nn.MSELoss()
    elif loss == 'huber':
        loss_func = nn.SmoothL1Loss()
    elif loss == 'perceptual':
        loss_func = PerceptualLoss(
            input_range='tanh',
            average_loss=False,
            extractor=encoder)
    elif loss == 'bce':
        loss_func = nn.BCEWithLogitsLoss()
    elif loss == 'ce':
        loss_func = nn.CrossEntropyLoss()

    criterion = lambda x, y: loss_func(x, y) * weight

    return criterion

def get_norm_layer(norm, dims=2):

    norm = norm.lower()

    if norm == 'batch':

        if dims == 1:
            norm_func = nn.BatchNorm1d
        elif dims == 2:
            norm_func = nn.BatchNorm2d
        elif dims == 3:
            norm_func = nn.BatchNorm3d

        norm_layer = lambda num_features: norm_func(
            num_features, 
            affine=True)

    elif norm == 'instance':

        if dims == 1:
            norm_func = nn.InstanceNorm1d
        elif dims == 2:
            norm_func = nn.InstanceNorm2d
        elif dims == 3:
            norm_func = nn.InstanceNorm3d

        norm_layer = lambda num_features: norm_func(
            num_features,
            affine=True,
            track_running_stats=False)

    elif norm == 'l1' or norm == 'l2':
        
        norm_layer = lambda num_features: Normalize(num_features, norm)

    elif norm == 'none':
        
        norm_layer = Identity

    return norm_layer

def get_nonlinear_layer(nonlinearity):

    nonlinearity = nonlinearity.lower()

    if nonlinearity == 'relu':
        nonlinear_layer = nn.ReLU
    elif nonlinearity == 'leakyrelu':
        nonlinear_layer = lambda inplace: nn.LeakyReLU(0.2, inplace)
    elif nonlinearity == 'swish':
        nonlinear_layer = Swish
    elif nonlinearity == 'tanh':
        nonlinear_layer = lambda inplace: nn.Tanh()
    elif nonlinearity == 'none':
        nonlinear_layer = lambda inplace: Identity()
    elif nonlinearity == 'threshold':
        nonlinear_layer = lambda inplace: Threshold(0.08726449, 0)

    return nonlinear_layer

def get_linear_block(
    in_channels, 
    out_channels, 
    nonlinear_layer=nn.ReLU,
    norm_layer=None,
    sequential=False,
    bias=None):

    # Prepare layers and parameters
    norm_layer = Identity if norm_layer is None else norm_layer
    bias = norm_layer == Identity if bias is None else bias

    layers = [
        nn.Linear(
            in_channels, 
            out_channels, 
            bias=bias),
        norm_layer(out_channels),
        nonlinear_layer(True)]

    if sequential:
        block = nn.Sequential(*layers)
    else:
        block = layers

    return block

def get_conv_block(
    in_channels, 
    out_channels, 
    nonlinear_layer=nn.ReLU,
    norm_layer=None,
    mode='same',
    sequential=False,
    kernel_size=3,
    factor=2,
    bias=None):

    # Prepare layers and parameters
    norm_layer = Identity if norm_layer is None else norm_layer
    bias = norm_layer == Identity if bias is None else bias

    stride = factor if mode == 'down' else 1
    assert stride != 1 or kernel_size % 2, 'Wrong kernel size for stride == 1'
    padding = (kernel_size - (stride > 1)) // 2

    layers = [
        nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride, 
            padding, 
            bias=bias),
        norm_layer(out_channels),
        nonlinear_layer(True)]

    if sequential:
        block = nn.Sequential(*layers)
    else:
        block = layers

    return block

def get_upsampling_block(
    in_channels,
    out_channels,
    nonlinear_layer=nn.ReLU,
    norm_layer=None,
    mode='conv_transpose',
    sequential=False,
    kernel_size=3,
    num_layers=1,
    factor=2,
    bias=None):

    # Prepare layers and parameters
    norm_layer = Identity if norm_layer is None else norm_layer
    bias = norm_layer == Identity if bias is None else bias

    stride = factor if mode == 'conv_transpose' else 1
    kernel_size = max(kernel_size, stride)
    check = kernel_size == 3 or kernel_size == 4
    assert check, 'kernel_size must be either 3 or 4'

    if mode == 'conv_transpose':

        padding = int(kernel_size != stride)
        output_padding = 1 if kernel_size == 3 else 0

        layers = [
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size, 
                stride, 
                padding,
                output_padding,
                bias=bias)]
    else:
        
        if mode == 'nearest':
            layers = [
                nn.Upsample(
                    scale_factor=factor, 
                    mode=mode)]
        else:
            layers = [
                nn.Upsample(
                    scale_factor=factor, 
                    mode=mode,
                    align_corners=False)]

        padding = 1

        layers += [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=bias)]

    layers += [
        norm_layer(out_channels),
        nonlinear_layer(True)]

    for i in range(num_layers-1):
        layers += get_conv_block(
            in_channels=out_channels,
            out_channels=out_channels,
            nonlinear_layer=nonlinear_layer,
            norm_layer=norm_layer,
            mode='same',
            sequential=False,
            kernel_size=kernel_size - (not kernel_size%2))

    if sequential:
        block = nn.Sequential(*layers)
    else:
        block = layers

    return block

def weights_init(module):
    """ Custom weights initialization """

    classname = module.__class__.__name__

    if classname.find('Conv') != -1:
        module.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.fill_(0)

class Swish(nn.Module):

    def __init__(self, inplace=True):
        super(Swish, self).__init__()

    def forward(self, input):
        return input * f.sigmoid(input)

    def __repr__(self):
        return ('{name}()'.format(name=self.__class__.__name__))


class Normalize(nn.Module):

    def __init__(self, num_channels=None, norm='l2'):
        super(Normalize, self).__init__()

        self.norm = norm

    def forward(self, input):

        if self.norm == 'l2':
            norm = input**2
        elif self.norm == 'l1':
            norm = np.abs(input)

        norm = norm.reshape(norm.shape[0], -1).sum(1)

        if self.norm == 'l2':
            norm = norm**0.5

        return input / norm[:, None, None, None].expand_as(input)

    def __repr__(self):
        return ('{name}(norm={norm})'.format(
            name=self.__class__.__name__,
            norm=self.norm))


class Identity(nn.Module):

    def __init__(self, num_channels=None):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

    def __repr__(self):
        return ('{name}()'.format(name=self.__class__.__name__))


class Threshold(nn.Module):

    def __init__(self, threshold, value):
        super(Threshold, self).__init__()

        self.threshold = threshold
        self.value = value

    def forward(self, input):

        input[input < self.threshold] = self.value

        return input


class View(nn.Module):

    def __init__(self):
        super(View, self).__init__()

    def forward(self, x, size=None):

        if len(x.shape) == 2:

            if size is None:
                return x.view(x.shape[0], -1, 1, 1)
            else:
                b, c = x.shape
                _, _, h, w = size
                return x.view(b, c, 1, 1).expand(b, c, h, w)

        elif len(x.shape) == 4:

            return x.view(x.shape[0], -1)

    def __repr__(self):
        return '{name}()'.format(name=self.__class__.__name__)


class ResBlock(nn.Module):
    
    def __init__(
        self, 
        in_channels, 
        nonlinear_layer=nn.ReLU,
        norm_layer=None):
        super(ResBlock, self).__init__()

        norm_layer = Identity if norm_layer is None else norm_layer
        bias = norm_layer == Identity

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=bias),
            norm_layer(in_channels),
            nonlinear_layer(True),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=bias),
            norm_layer(in_channels))

    def forward(self, input):

        return input + self.block(input)


class LocalAttention(nn.Module):
    
    def __init__(
        self, 
        in_channels,
        nonlinear_layer=nn.ReLU,
        norm_layer=None,
        num_channels=64,
        stride=32):
        super(LocalAttention, self).__init__()

        norm_layer = Identity if norm_layer is None else norm_layer
        bias = norm_layer == Identity

        self.num_channels = num_channels
        self.num_heads = in_channels // num_channels // 4 * 2

        out_channels = num_channels * 3 * self.num_heads

        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=bias),
            norm_layer(out_channels),
            nonlinear_layer(True))

        out_channels = num_channels * self.num_heads

        self.block_2 = nn.Sequential(
            nn.Conv2d(out_channels, in_channels, 3, 1, 1, bias=bias),
            norm_layer(in_channels))

        self.stride = stride
        
    def forward(self, input):
        
        output = self.block_1(input)
        
        b, c, h, w = output.shape
        nh, nw = h // self.stride, w // self.stride
        
        output = torch.cat(output.split(self.stride, dim=2), 0)
        output = torch.cat(output.split(self.stride, dim=3), 0)
        output = output.reshape(b*nh*nw, c, self.stride**2)
                
        keys, values, querries = output.split(c//3, dim=1)

        keys = torch.cat(keys.split(self.num_channels, dim=1), dim=0)
        values = torch.cat(values.split(self.num_channels, dim=1), dim=0)
        querries = torch.cat(querries.split(self.num_channels, dim=1), dim=0)

        attention = torch.bmm(keys.transpose(1, 2), querries)
        attention = f.softmax(attention / (self.num_channels)**0.5, dim=1)
        
        output = torch.bmm(values, attention)

        output = torch.cat(output.split(b*nh*nw, dim=0), dim=1)
        output = output.reshape(b*nh*nw, c//3, self.stride, self.stride)
        output = torch.cat(output.split(b*nh, dim=0), dim=3)
        output = torch.cat(output.split(b, dim=0), dim=2)
        
        return input + self.block_2(output)


class ConcatBlock(nn.Module):

    def __init__(
        self,
        enc_channels,
        out_channels, 
        nonlinear_layer=nn.ReLU,
        norm_layer=None,
        norm_layer_cat=None,
        kernel_size=3):
        super(ConcatBlock, self).__init__()

        norm_layer = Identity if norm_layer is None else norm_layer
        norm_layer_cat = Identity if norm_layer_cat is None else norm_layer_cat

        # Get branch from encoder
        layers = get_conv_block(
                enc_channels,
                out_channels,
                nonlinear_layer,
                norm_layer,
                'same', False,
                kernel_size)
        
        layers += [norm_layer_cat(out_channels)]

        self.enc_block = nn.Sequential(*layers)
#        self.dis_block = norm_layer_cat(out_channels)

    def forward(self, input, vgg_input):

        output_enc = self.enc_block(vgg_input)
#        output_dis = self.dis_block(input)
        output_dis = input

        output = torch.cat([output_enc, output_dis], 1)

        return output

def save_checkpoint(model, prefix):

    prefix = str(prefix)

    if hasattr(model, 'gen_A'):
        torch.save(
            model.gen_A.module.cpu(),
            os.path.join(model.weights_path, '%s_gen_A.pkl' % prefix))
        model.gen_A.module.cuda(model.gpu_id)

    if hasattr(model, 'gen_B'):
        torch.save(
            model.gen_B.module.cpu(),
            os.path.join(model.weights_path, '%s_gen_B.pkl' % prefix))
        model.gen_B.module.cuda(model.gpu_id)
    
    if hasattr(model, 'dis_A'):
        torch.save(
            model.dis_A.module.cpu(),
            os.path.join(model.weights_path, '%s_dis_A.pkl' % prefix))
        model.dis_A.module.cuda(model.gpu_id)
    
    if hasattr(model, 'dis_B'):
        torch.save(
            model.dis_B.module.cpu(),
            os.path.join(model.weights_path, '%s_dis_B.pkl' % prefix))
        model.dis_B.module.cuda(model.gpu_id)

def load_checkpoint(model, prefix, path=''):

    prefix = str(prefix)

    print('\nLoading checkpoint %s from path %s' % (prefix, path))

    if not path: path = model.weights_path

    path_gen_A = os.path.join(path, '%s_gen_A.pkl' % prefix)
    path_gen_B = os.path.join(path, '%s_gen_B.pkl' % prefix)

    if hasattr(model, 'gen_A'):
        if os.path.exists(path_gen_A):
            model.gen_A = torch.load(path_gen_A)
        elif os.path.exists(path_gen_B):
            model.gen_A = torch.load(path_gen_B)

    if hasattr(model, 'gen_B'):
        if os.path.exists(path_gen_B):
            model.gen_B = torch.load(path_gen_B)
        elif os.path.exists(path_gen_A):
            model.gen_B = torch.load(path_gen_A)

    path_dis_A = os.path.join(path, '%s_dis_A.pkl' % prefix)
    path_dis_B = os.path.join(path, '%s_dis_B.pkl' % prefix)

    if hasattr(model, 'dis_A'):
        if os.path.exists(path_dis_A):
            model.dis_A = torch.load(path_dis_A)
        elif os.path.exists(path_dis_B):
            model.dis_A = torch.load(path_dis_B)

    if hasattr(model, 'dis_B'):
        if os.path.exists(path_dis_B):
            model.dis_B = torch.load(path_dis_B)
        elif os.path.exists(path_dis_A):
            model.dis_B = torch.load(path_dis_A)

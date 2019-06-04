import sys
sys.path += ['',
 '/mnt/mlhep2018/pythia/lib',
 '/home/rkhairulin/.local/lib/python3.6/site-packages',
 '/mnt/mlhep2018/pyenv/versions/3.6.6/lib/python36.zip',
 '/mnt/mlhep2018/pyenv/versions/3.6.6/lib/python3.6',
 '/mnt/mlhep2018/pyenv/versions/3.6.6/lib/python3.6/lib-dynload',
 '/mnt/mlhep2018/pyenv/versions/3.6.6/envs/mlhep/lib/python3.6/site-packages',
 '/mnt/mlhep2018/pyenv/versions/3.6.6/envs/mlhep/lib/python3.6/site-packages/IPython/extensions',
 '/home/rkhairulin/.ipython',
 '/home/rkhairulin/perceptual_gan_dev/',
 '/home/rkhairulin/perceptual_gan_dev/',
 '/home/rkhairulin/perceptual_gan_dev/']
import argparse
import importlib
from models.discriminator import get_args as get_args_dis
from src.dataset import get_dataloaders
from src.logs import Logs
from src import utils
import os
import numpy as np
import torch
from torch.optim import Adam
import nirvana_dl

parser = argparse.ArgumentParser(conflict_handler='resolve')
parser.add = parser.add_argument

# Path to txt list or npy or path to folder with images
parser.add('--train_img_A_path', default='', type=str)
parser.add('--train_img_B_path', default='', type=str)
parser.add('--test_img_A_path', default='', type=str)
parser.add('--test_img_B_path', default='', type=str)

# Path to npy with data
parser.add('--train_aux_A_path', default='', type=str)
parser.add('--train_aux_B_path', default='', type=str)
parser.add('--test_aux_A_path', default='', type=str)
parser.add('--test_aux_B_path', default='', type=str)

# Dataset opts
parser.add('--images_path', default='', type=str,
           help='required when train/test_A/B_path options are txt lists')
parser.add('--num_workers', default=4, type=int,
           help='number of data loading workers')
parser.add('--batch_size', default=16, type=int,
           help='batch size')
parser.add('--input_transforms', default='', type=str,
           help='scale|crop|augment')

parser.add('--img_size', default=0  , type=int,
           help='the resolution of the input image to network')
parser.add('--img_channels', default=3, type=int,
           help='dimensionality of the image')

# Opts for auxiliary inputs
parser.add('--aux_channels', default=0, type=int)
parser.add('--aux_loss_type', default='huber', type=str,
           help='bce|ce|mse|l1|huber')
parser.add('--gen_aux_loss_weight', default=0., type=float)
parser.add('--dis_aux_loss_weight', default=0., type=float)
parser.add('--pretrained_aux_path', default='', type=str)

# If general opts are specified, these options are overwritten
parser.add('--img_size_A', default=0, type=int)
parser.add('--img_size_B', default=0, type=int)
parser.add('--img_channels_A', default=0, type=int)
parser.add('--img_channels_B', default=0, type=int)
parser.add('--aux_channels_A', default=0, type=int)
parser.add('--aux_channels_B', default=0, type=int)

# Training opts
parser.add('--gpu_ids', default='0', type=str)
parser.add('--epoch_len', default=100, type=int,
           help='number of batches per epoch')
parser.add('--num_epoch', default=200, type=int,
           help='number of epochs to train for')
parser.add('--lr', default=1e-4, type=float,
           help='learning rate, default=1e-4')
parser.add('--beta1', default=0.9, type=float,
           help='beta1 for adam. default=0.9')
parser.add('--schedule_freq', default=1, type=int,
           help='how many dis updates for one gen update')
parser.add('--manual_seed', default=9107, type=int,
           help='set random seed')
parser.add('--experiment_dir', default='runs')
parser.add('--experiment_name', default='')
parser.add('--experiment_path', default='')
parser.add('--save_every_epoch', default=1, type=int)
parser.add('--which_epoch', type=str, default='latest',
           help='continue train from which epoch')

# Generator opts
parser.add('--gen_type_names', default='', type=str,
           help='specify generator type names separated by commas')

# If gen opt is specified, these options are overwritten
parser.add('--gen_type_name_A', default='', type=str,
           help='specify generator type name')
parser.add('--gen_type_name_B', default='', type=str,
           help='specify generator type name')

# Discriminator opts
parser.add('--dis_type_names', default='', type=str,
           help='specify discriminator type names separated by commas')
parser.add('--dis_down_fracs', default='1', type=str,
           help='downsampling factor for each of these discriminators')

# If dis opts are specified, these options are overwritten
parser.add('--dis_type_names_A', default='', type=str,
           help='specify discriminator type names separated by commas')
parser.add('--dis_down_fracs_A', default='', type=str,
           help='downsampling factor for each of these discriminators')
parser.add('--dis_type_names_B', default='', type=str,
           help='specify discriminator type names separated by commas')
parser.add('--dis_down_fracs_B', default='', type=str,
           help='downsampling factor for each of these discriminators')

# Shared opts between generator and discriminator
parser.add('--num_channels', default=0, type=int)
parser.add('--max_channels', default=0, type=int)
parser.add('--model_type', default='cyclegan', type=str,
           help='cyclegan|pix2pix|dcgan')
parser.add('--enc_type', default='vgg19_pytorch_modified', type=str)
parser.add('--mse_loss_type', default='', type=str)
parser.add('--mse_loss_weight', default=0., type=float)
parser.add('--pretrained_gen_path', default='', type=str)

# If dis opts are specified, these options are overwritten
parser.add('--mse_loss_type_A', default='', type=str)
parser.add('--mse_loss_type_B', default='', type=str)
parser.add('--mse_loss_weight_A', default=0., type=float)
parser.add('--mse_loss_weight_B', default=0., type=float)

# Opts for unaligned or aligned image translation
opt, _ = parser.parse_known_args()

# Set seed
#os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids[0]
np.random.seed(opt.manual_seed)
torch.manual_seed(opt.manual_seed)
torch.cuda.manual_seed_all(opt.manual_seed)
torch.cuda.set_device(int(opt.gpu_ids[0]))

# Add options for generator types
gen_type_names = []
if opt.gen_type_names: gen_type_names += [opt.gen_type_names.split(',')]
if opt.gen_type_name_A: gen_type_names += [opt.gen_type_name_A.split(',')]
if opt.gen_type_name_B: gen_type_names += [opt.gen_type_name_B.split(',')]
gen_type_names = np.unique(gen_type_names)

# Import correct generator model
if opt.model_type == 'dcgan':
    m = importlib.import_module('models.generator')
else:
    m = importlib.import_module('models.translation_generator')

for type_name in gen_type_names:
    m.get_args(parser, type_name)

# Add options for discriminator types
dis_type_names = []
if opt.dis_type_names: dis_type_names += [opt.dis_type_names.split(',')]
if opt.dis_type_names_A: dis_type_names += [opt.dis_type_names_A.split(',')]
if opt.dis_type_names_B: dis_type_names += [opt.dis_type_names_B.split(',')]
dis_type_names = np.unique(dis_type_names)

for type_name in dis_type_names:
    get_args_dis(parser, type_name)

# Parse new arguments
opt, _ = parser.parse_known_args()

# Experiment dir
opt.experiment_path = os.path.join(
    opt.experiment_dir,
    opt.experiment_name)

# Fill in options for domains in case its needed
if opt.img_size: 
    opt.img_size_A, opt.img_size_B = [opt.img_size]*2
if opt.img_channels: 
    opt.img_channels_A, opt.img_channels_B = [opt.img_channels]*2
if opt.aux_channels: 
    opt.aux_channels_A, opt.aux_channels_B = [opt.aux_channels]*2
if opt.gen_type_names:
    opt.gen_type_name_A, opt.gen_type_name_B = [opt.gen_type_names]*2
if opt.dis_type_names:
    opt.dis_type_names_A, opt.dis_type_names_B = [opt.dis_type_names]*2
if opt.dis_down_fracs:
    opt.dis_down_fracs_A, opt.dis_down_fracs_B = [opt.dis_down_fracs]*2
if opt.mse_loss_type:
    opt.mse_loss_type_A, opt.mse_loss_type_B = [opt.mse_loss_type]*2
if opt.mse_loss_weight:
    opt.mse_loss_weight_A, opt.mse_loss_weight_B = [opt.mse_loss_weight]*2

print(opt)

# Make directories
if not os.path.exists(opt.experiment_path):
    os.makedirs(opt.experiment_path)
    os.makedirs(opt.experiment_path + '/checkpoints')

# Save opts
file_name = opt.experiment_path + '/opt.txt'
with open(file_name, 'wt') as opt_file:
    for k, v in sorted(vars(opt).items()):
        opt_file.write('%s: %s\n' % (str(k), str(v)))

# Preprocess the options
opt.input_transforms = opt.input_transforms.split(',')
opt.gpu_ids = [int(i) for i in opt.gpu_ids.split(',')]
opt.dis_type_names_A = opt.dis_type_names_A.split(',')
opt.dis_type_names_B = opt.dis_type_names_B.split(',')
opt.dis_down_fracs_A = [int(i) for i in opt.dis_down_fracs_A.split(',')]
opt.dis_down_fracs_B = [int(i) for i in opt.dis_down_fracs_B.split(',')]

# Get dataloaders
train_dataloader, test_dataloader = get_dataloaders(opt)

# Initialize model
m = importlib.import_module('models.' + opt.model_type)
model = m.Model(opt)

# Initialize optimizers
if hasattr(model, 'gen_params'):

    opt_G = Adam(model.gen_params, lr=opt.lr, betas=(opt.beta1, 0.999))

    path_opt_G = os.path.join(
        model.weights_path, 
        '%s_opt_G.pkl' % opt.which_epoch)

    if os.path.exists(path_opt_G):
        opt_G.load_state_dict(torch.load(path_opt_G))

if hasattr(model, 'dis_params'):

    opt_D = Adam(model.dis_params, lr=opt.lr, betas=(opt.beta1, 0.999))

    path_opt_D = os.path.join(
        model.weights_path, 
        '%s_opt_D.pkl' % opt.which_epoch)

    if os.path.exists(path_opt_D):
        opt_D.load_state_dict(torch.load(path_opt_D))

logs = Logs(model, opt)

epoch_start = 0 if opt.which_epoch == 'latest' else int(opt.which_epoch)

for epoch in range(epoch_start + 1, opt.num_epoch + 1):
    print("NOW EPOCH IS ", epoch)
    model.train()

    schedule_iter = 1

    for inputs in train_dataloader:

        schedule_iter %= opt.schedule_freq

        model.forward(inputs)

        if hasattr(model, 'gen_params') and not schedule_iter:

            if hasattr(model, 'dis_params'):

                for p in model.dis_params:
                    p.requires_grad = False

            opt_G.zero_grad()
            model.backward_G()
            opt_G.step()

            if hasattr(model, 'dis_params'):

                for p in model.dis_params:
                    p.requires_grad = True

        if hasattr(model, 'dis_params'):

            opt_D.zero_grad()
            model.backward_D()
            opt_D.step()
            
        logs.update_losses('train')

        schedule_iter += 1

    model.eval()

    for inputs in test_dataloader:

        with torch.no_grad():

            model.forward(inputs)
            
            if hasattr(model, 'gen_params'): model.backward_G()
            if hasattr(model, 'dis_params'): model.backward_D()
            
        logs.update_losses('test')

    logs.update_tboard(epoch)
    
    # Save weights
    if not epoch % opt.save_every_epoch:
        print("SAVING PARAMETERS")
        utils.save_checkpoint(model, epoch)

        if hasattr(model, 'gen_params'):
            torch.save(
                opt_G.state_dict(),
                os.path.join(model.weights_path, '%d_opt_G.pkl' % epoch))

        if hasattr(model, 'dis_params'):
            torch.save(
                opt_D.state_dict(),
                os.path.join(model.weights_path, '%d_opt_D.pkl' % epoch))
        nirvana_dl.snapshot.dump_snapshot()

logs.close()
utils.save_checkpoint(model, 'latest')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import numpy as np
import copy
import sys
sys.path.append(('../'))
from util import swap_axis
sys.path.append('../../')
from FKP.network import KernelPrior

'''
# ------------------------------------------------
# networks of original KernelGAN and KernelGAN-KP
# ------------------------------------------------
'''

# ----------------------------------
# Networks of original KernelGAN
# ----------------------------------
class Generator(nn.Module):
    ''' Generator of original KernelGAN '''

    def __init__(self, conf):
        super(Generator, self).__init__()
        struct = conf.G_structure
        # First layer - Converting RGB image to latent space
        self.first_layer = nn.Conv2d(in_channels=1, out_channels=conf.G_chan, kernel_size=struct[0], bias=False)

        feature_block = []  # Stacking intermediate layer
        for layer in range(1, len(struct) - 1):
            feature_block += [
                nn.Conv2d(in_channels=conf.G_chan, out_channels=conf.G_chan, kernel_size=struct[layer], bias=False)]
        self.feature_block = nn.Sequential(*feature_block)
        # Final layer - Down-sampling and converting back to image
        self.final_layer = nn.Conv2d(in_channels=conf.G_chan, out_channels=1, kernel_size=struct[-1],
                                     stride=int(1 / conf.scale_factor), bias=False)

        # Calculate number of pixels shaved in the forward pass
        self.output_size = \
            self.forward(torch.FloatTensor(torch.ones([1, 3, conf.input_crop_size, conf.input_crop_size]))).shape[-1]
        self.forward_shave = int(conf.input_crop_size * conf.scale_factor) - self.output_size

    def forward(self, input_tensor):
        # Swap axis of RGB image for the network to get a "batch" of size = 3 rather the 3 channels
        input_tensor = swap_axis(input_tensor)
        downscaled = self.first_layer(input_tensor)
        features = self.feature_block(downscaled)
        output = self.final_layer(features)
        return swap_axis(output)


class Discriminator(nn.Module):
    ''' Discriminator of original KernelGAN '''

    def __init__(self, conf):
        super(Discriminator, self).__init__()

        # First layer - Convolution (with no ReLU)
        self.first_layer = nn.utils.spectral_norm(
            nn.Conv2d(in_channels=3, out_channels=conf.D_chan, kernel_size=conf.D_kernel_size, bias=True))
        feature_block = []  # Stacking layers with 1x1 kernels
        for _ in range(1, conf.D_n_layers - 1):
            feature_block += [nn.utils.spectral_norm(
                nn.Conv2d(in_channels=conf.D_chan, out_channels=conf.D_chan, kernel_size=1, bias=True)),
                nn.BatchNorm2d(conf.D_chan),
                nn.ReLU(True)]
        self.feature_block = nn.Sequential(*feature_block)
        self.final_layer = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels=conf.D_chan, out_channels=1, kernel_size=1, bias=True)),
            nn.Sigmoid())

        # Calculate number of pixels shaved in the forward pass
        self.forward_shave = conf.input_crop_size - self.forward(
            torch.FloatTensor(torch.ones([1, 3, conf.input_crop_size, conf.input_crop_size]))).shape[-1]

    def forward(self, input_tensor):
        receptive_extraction = self.first_layer(input_tensor)
        features = self.feature_block(receptive_extraction)
        return self.final_layer(features)


def weights_init_D(m):
    """ initialize weights of the discriminator """
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight, 0.1)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif class_name.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def weights_init_G(m):
    """ initialize weights of the generator """
    if m.__class__.__name__.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight, 0.1)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)


# -------------------------
# Networks of KernelGAN-KP
# -------------------------
class Generator_KP(nn.Module):
    ''' Generator of KernelGAN-KP '''

    def __init__(self, conf):
        super(Generator_KP, self).__init__()

        # initialze the kernel to be smooth is slightly better
        seed = 1
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = True

        self.kernel_size = conf.G_kernel_size
        self.scale = int(1 / conf.scale_factor)

        # load kernel prior
        self.kp = KernelPrior(n_blocks=5, input_size=self.kernel_size ** 2, hidden_size=15, n_hidden=1)
        state = torch.load(conf.path_KP)
        self.kp.load_state_dict(state['model_state'])
        self.kp.eval()
        for p in self.kp.parameters(): p.requires_grad = False

        # random initialize latent variable
        self.kernel_code = nn.Parameter(self.kp.base_dist.sample((1, 1)), requires_grad=True)

        # Calculate number of pixels shaved in the forward pass
        self.output_size = \
            self.forward(torch.FloatTensor(torch.ones([1, 3, conf.input_crop_size, conf.input_crop_size])))[0].shape[-1]
        self.forward_shave = int(conf.input_crop_size * conf.scale_factor) - self.output_size

    def forward(self, input_tensor):
        # generate kernel
        out_k, logprob = self.kp.inverse(self.kernel_code)
        out_k = self.kp.post_process(out_k)

        # blur
        out_put = F.conv2d(input_tensor, out_k.expand(3, -1, -1, -1), groups=3)

        # downscale
        out_put = out_put[:, :, 0::self.scale, 0::self.scale]

        return out_put, out_k.squeeze()


class Discriminator_KP(nn.Module):
    ''' Discriminator of KernelGAN-KP '''

    def __init__(self, conf, d_input_shape=27):
        super(Discriminator_KP, self).__init__()

        self.head = ConvBlock(3, conf.D_chan, conf.D_kernel_size, 0, 1)

        self.body = nn.Sequential()
        for i in range(conf.D_n_layers - 2):
            block = ConvBlock(conf.D_chan, conf.D_chan, conf.D_kernel_size, 0, 1)
            self.body.add_module('block%d' % (i + 1), block)

        self.tail = nn.Conv2d(conf.D_chan, 1, kernel_size=conf.D_kernel_size, stride=1, padding=0)

        # Calculate number of pixels shaved in the forward pass
        self.output_size = self.forward(torch.FloatTensor(torch.ones([1, 3, d_input_shape, d_input_shape]))).shape[-1]

    def forward(self, x):
        x = x * 2 - 1
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x


class ConvBlock(nn.Sequential):
    ''' convblock used in Discriminator_KP'''

    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock, self).__init__()
        self.add_module('conv',
                        (nn.Conv2d(in_channel, out_channel, kernel_size=ker_size, stride=stride, padding=padd))),
        self.add_module('norm', nn.BatchNorm2d(out_channel)),
        self.add_module('LeakyRelu', nn.LeakyReLU(0.2, inplace=True))


def weights_init_G_KP(m):
    """ initialize weights of the generator_KP """
    if m.__class__.__name__.find('Conv') != -1 and m.__class__.__name__.find('nf') == -1:
        nn.init.xavier_normal_(m.weight, 0.1)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)


def weights_init_D_KP(m):
    """ initialize weights of the discriminator_KP """
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

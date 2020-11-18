import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import tqdm
import os
import matplotlib.pyplot as plt
from .networks import skip, fcn
from .SSIM import SSIM

sys.path.append('../')
from util import save_final_kernel_png, get_noise, kernel_shift, move2cpu, tensor2im01

sys.path.append('../../')
from FKP.network import KernelPrior

'''
# ------------------------------------------
# models of DIPFKP, etc.
# ------------------------------------------
'''


class DIPFKP:
    '''
    # ------------------------------------------
    # (1) create model, loss and optimizer
    # ------------------------------------------
    '''

    def __init__(self, conf, lr, device=torch.device('cuda')):

        # Acquire configuration
        self.conf = conf
        self.lr = lr
        self.sf = conf.sf
        self.kernel_size = min(conf.sf * 4 + 3, 21)

        # DIP model
        _, C, H, W = self.lr.size()
        self.input_dip = get_noise(C, 'noise', (H * self.sf, W * self.sf)).to(device).detach()
        self.net_dip = skip(C, 3,
                            num_channels_down=[128, 128, 128, 128, 128],
                            num_channels_up=[128, 128, 128, 128, 128],
                            num_channels_skip=[16, 16, 16, 16, 16],
                            upsample_mode='bilinear',
                            need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU')
        self.net_dip = self.net_dip.to(device)
        self.optimizer_dip = torch.optim.Adam([{'params': self.net_dip.parameters()}], lr=conf.dip_lr)

        # normalizing flow as kernel prior
        if conf.model == 'DIPFKP':
            # initialze the kernel to be smooth is slightly better
            seed = 5
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.benchmark = True

            self.net_kp = KernelPrior(n_blocks=5, input_size=self.kernel_size ** 2, hidden_size=min((self.sf+1)*5, 25),
                                      n_hidden=1)

            state = torch.load(conf.path_KP)
            self.net_kp.load_state_dict(state['model_state'])
            self.net_kp = self.net_kp.to(device)
            self.net_kp.eval()
            for p in self.net_kp.parameters(): p.requires_grad = False

            self.kernel_code = self.net_kp.base_dist.sample((1, 1)).to(device)
            self.kernel_code.requires_grad = True

            self.optimizer_kp = SphericalOptimizer(self.kernel_size, torch.optim.Adam, [self.kernel_code],
                                                   lr=conf.kp_lr)

        # baseline, softmax as kernel prior
        elif conf.model == 'DIPSoftmax':
            self.kernel_code =torch.ones(self.kernel_size ** 2).to(device)
            self.kernel_code.requires_grad = True

            self.optimizer_kp = torch.optim.Adam([{'params': self.kernel_code}], lr=conf.kp_lr)

        # fc layers as kernel prior, accroding to Double-DIP/Selfdeblur, set lr = 1e-4
        elif conf.model == 'DoubleDIP':
            n_k = 200
            self.kernel_code = get_noise(n_k, 'noise', (1, 1)).detach().squeeze().to(device)

            self.net_kp = fcn(n_k, self.kernel_size ** 2).to(device)

            self.optimizer_kp = torch.optim.Adam([{'params': self.net_kp.parameters()}], lr=1e-4)

        # loss
        self.ssimloss = SSIM().to(device)
        self.mse = torch.nn.MSELoss().to(device)

        print('*' * 60 + '\nSTARTED {} on: {}...'.format(conf.model, conf.input_image_path))

    '''
    # ---------------------
    # (2) training
    # ---------------------
    '''

    def train(self):
        for iteration in tqdm.tqdm(range(self.conf.max_iters), ncols=60):
            iteration += 1

            self.optimizer_dip.zero_grad()
            if self.conf.model == 'DIPFKP':
                self.optimizer_kp.opt.zero_grad()
            else:
                self.optimizer_kp.zero_grad()

            '''
            # ---------------------
            # (2.1) forward
            # ---------------------
             '''

            # generate sr image
            sr = self.net_dip(self.input_dip)

            # generate kernel
            if self.conf.model == 'DIPFKP':
                kernel, logprob = self.net_kp.inverse(self.kernel_code)
                kernel = self.net_kp.post_process(kernel)
            elif self.conf.model == 'DIPSoftmax':
                kernel = torch.softmax(self.kernel_code, 0).view(1, 1, self.kernel_size, self.kernel_size)
            elif self.conf.model == 'DoubleDIP':
                kernel = self.net_kp(self.kernel_code).view(1, 1, self.kernel_size, self.kernel_size)

            # blur
            sr_pad = F.pad(sr, mode='circular',
                       pad=(self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2))
            out = F.conv2d(sr_pad, kernel.expand(3, -1, -1, -1), groups=3)

            # downscale
            out = out[:, :, 0::self.sf, 0::self.sf]

            '''
            # ---------------------
            # (2.2) backward
            # ---------------------
             '''
            # freeze kernel estimation, so that DIP can train first to learn a meaningful image
            if iteration <= 75:
                self.kernel_code.requires_grad = False
            else:
                self.kernel_code.requires_grad = True

            # first use SSIM because it helps the model converge faster
            if iteration <= 80:
                loss = 1 - self.ssimloss(out, self.lr)
            else:
                loss = self.mse(out, self.lr)

            loss.backward()
            self.optimizer_dip.step()
            self.optimizer_kp.step()

            if (iteration % 10 == 0 or iteration == 1) and self.conf.verbose:
                save_final_kernel_png(move2cpu(kernel.squeeze()), self.conf, self.conf.kernel_gt, iteration)
                plt.imsave(os.path.join(self.conf.output_dir_path, '{}_{}.png'.format(self.conf.img_name, iteration)),
                                        tensor2im01(sr), vmin=0, vmax=1., dpi=1)
                print('\n Iter {}, loss: {}'.format(iteration, loss.data))

        kernel = move2cpu(kernel.squeeze())
        save_final_kernel_png(kernel, self.conf, self.conf.kernel_gt)

        if self.conf.verbose:
            print('{} estimation complete! (see --{}-- folder)\n'.format(self.conf.model,
                                                                         self.conf.output_dir_path) + '*' * 60 + '\n\n')

        return kernel, sr


class SphericalOptimizer(torch.optim.Optimizer):
    ''' spherical optimizer, optimizer on the sphere of the latent space'''

    def __init__(self, kernel_size, optimizer, params, **kwargs):
        self.opt = optimizer(params, **kwargs)
        self.params = params
        with torch.no_grad():
            # in practice, setting the radii as kernel_size-1 is slightly better
            self.radii = {param: torch.ones([1, 1, 1]).to(param.device) * (kernel_size - 1) for param in params}

    @torch.no_grad()
    def step(self, closure=None):
        loss = self.opt.step(closure)
        for param in self.params:
            param.data.div_((param.pow(2).sum(tuple(range(2, param.ndim)), keepdim=True) + 1e-9).sqrt())
            param.mul_(self.radii[param])

        return loss

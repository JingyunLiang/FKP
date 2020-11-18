import os
import argparse
import torch
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from util import read_image, im2tensor01, map2tensor, tensor2im01, analytic_kernel, kernel_shift, evaluation_dataset
from config.configs import Config
from model.model import DIPFKP

# for nonblind SR
sys.path.append('../')
from NonblindSR.usrnet import USRNet

'''
# ------------------------------------------------
# main.py for DIP-KP
# ------------------------------------------------
'''


def train(conf, lr_image):
    ''' trainer for DIPFKP, etc.'''
    model = DIPFKP(conf, lr_image)
    kernel, sr = model.train()
    return kernel, sr


def create_params(filename, args):
    ''' pass parameters to Config '''
    params = ['--model', args.model,
              '--input_image_path', os.path.join(args.input_dir, filename),
              '--output_dir_path', os.path.abspath(args.output_dir),
              '--path_KP', os.path.abspath(args.path_KP),
              '--sf', args.sf]
    if args.SR:
        params.append('--SR')
    if args.real:
        params.append('--real')
    return params


def main():
    # Parse the command line arguments
    prog = argparse.ArgumentParser()
    prog.add_argument('--model', type=str, default='DIPFKP', help='models: DIPFKP, DIPSoftmax, DoubleDIP.')
    prog.add_argument('--dataset', '-d', type=str, default='Set5',
                      help='dataset, e.g., Set5.')
    prog.add_argument('--sf', type=str, default='2', help='The wanted SR scale factor')
    prog.add_argument('--path-nonblind', type=str, default='../data/pretrained_models/usrnet_tiny.pth',
                      help='path for trained nonblind model')
    prog.add_argument('--SR', action='store_true', default=False, help='when activated - nonblind SR is performed')
    prog.add_argument('--real', action='store_true', default=False, help='if the input is real image')

    # to be overwritten automatically
    prog.add_argument('--path-KP', type=str, default='../data/pretrained_models/FKP_x2.pt',
                      help='path for trained kernel prior')
    prog.add_argument('--input-dir', '-i', type=str, default='../data/datasets/Set5/DIPFKP_lr_x2',
                      help='path to image input directory.')
    prog.add_argument('--output-dir', '-o', type=str,
                      default='../data/log_KernelGANFKP/Set5_DIPFKP_lr_x2', help='path to image output directory')


    args = prog.parse_args()

    # overwritting paths
    args.path_KP = '../data/pretrained_models/FKP_x{}.pt'.format(args.sf)
    args.input_dir = '../data/datasets/{}/DIPFKP_lr_x{}'.format(args.dataset, args.sf)
    args.output_dir = '../data/log_DIPFKP/{}_{}_lr_x{}'.format(args.dataset, args.model, args.sf)

    # load nonblind model
    if args.SR:
        netG = USRNet(n_iter=6, h_nc=32, in_nc=4, out_nc=3, nc=[16, 32, 64, 64],
                      nb=2, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")
        netG.load_state_dict(torch.load(args.path_nonblind), strict=True)
        netG.eval()
        for key, v in netG.named_parameters():
            v.requires_grad = False
        netG = netG.cuda()

    filesource = os.listdir(os.path.abspath(args.input_dir))
    filesource.sort()
    for filename in filesource[:]:
        print(filename)

        # kernel estimation
        conf = Config().parse(create_params(filename, args))
        lr_image = im2tensor01(read_image(os.path.join(args.input_dir, filename))).unsqueeze(0)

        # crop the image to 960x960 due to memory limit
        if 'DIV2K' in args.input_dir:
            crop = int(960 / 2 / conf.sf)
            lr_image = lr_image[:, :, lr_image.shape[2] // 2 - crop: lr_image.shape[2] // 2 + crop,
                       lr_image.shape[3] // 2 - crop: lr_image.shape[3] // 2 + crop]

        kernel, sr_dip = train(conf, lr_image)
        plt.imsave(os.path.join(conf.output_dir_path, '%s.png' % conf.img_name), tensor2im01(sr_dip), vmin=0,
                   vmax=1., dpi=1)

        # nonblind SR
        if args.SR:
            kernel = map2tensor(kernel)

            sr = netG(lr_image, torch.flip(kernel, [2, 3]), int(args.sf),
                      (10 if args.real else 0) / 255 * torch.ones([1, 1, 1, 1]).cuda())
            plt.imsave(os.path.join(conf.output_dir_path, '%s.png' % conf.img_name), tensor2im01(sr), vmin=0,
                       vmax=1., dpi=1)

    if not conf.verbose:
        evaluation_dataset(args.input_dir, conf)

    prog.exit(0)


if __name__ == '__main__':
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

    main()

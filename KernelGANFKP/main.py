import os
import argparse
import torch
import sys
import tqdm
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
from util import read_image, im2tensor01, map2tensor, tensor2im01, evaluation_dataset
# for KernelGAN-KP
from config.configs_FKP import Config_FKP
from dataloader.dataloader_FKP import DataGenerator_FKP
from model.model_FKP import KernelGAN_FKP
# for KernelGAN
from config.configs import Config
from dataloader.dataloader import DataGenerator
from model.model import KernelGAN
from model.learner import Learner

# for nonblind SR
sys.path.append('../')
from NonblindSR.usrnet import USRNet

'''
# ------------------------------------------------
# main.py for KernelGAN-KP
# ------------------------------------------------
'''


def train_FKP(conf):
    ''' trainer for KernelGAN-FKP'''
    gan = KernelGAN_FKP(conf)
    data = DataGenerator_FKP(conf, gan)
    data_loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True, num_workers=16, pin_memory=True)

    # X4 needs more iterations to converge
    total_iter = 4000 if conf.X4 else 1000
    for batch_idx, (g_in, d_in) in tqdm.tqdm(enumerate(data_loader), total=total_iter, ncols=60):
        if batch_idx >= total_iter:
            break
        gan.train(g_in, d_in, batch_idx + 1)

    kernel = gan.finish()
    return kernel


def train(conf):
    ''' trainer for KernelGAN'''
    gan = KernelGAN(conf)
    learner = Learner()
    data = DataGenerator(conf, gan)
    for iteration in tqdm.tqdm(range(conf.max_iters), ncols=60):
        [g_in, d_in] = data.__getitem__(iteration)
        gan.train(g_in, d_in, iteration)
        learner.update(iteration, gan)
    kernel = gan.finish()
    return kernel


def create_params(filename, args):
    ''' pass parameters to Config '''
    params = ['--input_image_path', os.path.join(args.input_dir, filename),
              '--output_dir_path', os.path.abspath(args.output_dir),
              '--path_KP', os.path.abspath(args.path_KP)]
    if args.X4:
        params.append('--X4')
    if args.SR:
        params.append('--SR')
    if args.real:
        params.append('--real')
    return params


def main():
    # Parse the command line arguments
    prog = argparse.ArgumentParser()
    prog.add_argument('--model', type=str, default='KernelGANFKP', help='KernelGANFKP or KernelGAN.')
    prog.add_argument('--dataset', '-d', type=str, default='../data/datasets/DIV2K/KernelGANFKP_lr_x2',
                      help='dataset, e.g., DIV2K.')
    prog.add_argument('--sf', type=str, default='2',
                      help='The wanted SR scale factor, KernelGAN only supports 2 or 4.')
    prog.add_argument('--SR', action='store_true', default=False, help='when activated - nonblind SR is performed')
    prog.add_argument('--real', action='store_true', default=False,
                      help='if the input is real image, to be overwritten automatically.')
    prog.add_argument('--path-KP', type=str, default='../data/pretrained_models/FKP_x2.pt',
                      help='path for trained kernel prior')
    prog.add_argument('--path-nonblind', type=str, default='../data/pretrained_models/usrnet_tiny.pth',
                      help='path for trained nonblind model')

    # to be overwritten automatically
    prog.add_argument('--input-dir', '-i', type=str, default='../data/datasets/DIV2K/KernelGANFKP_lr_x2',
                      help='path to image input directory, to be overwritten automatically.')
    prog.add_argument('--output-dir', '-o', type=str,
                      default='../data/log_KernelGANFKP/DIV2K_KernelGANFKP_lr_x2',
                      help='path to image output directory, to be overwritten automatically.')
    prog.add_argument('--X4', action='store_true', default=False,
                      help='The wanted SR scale factor, to be overwritten automatically.')

    args = prog.parse_args()

    # overwritting paths
    if args.sf == '2':
        args.X4 = False
    elif args.sf == '4':
        args.X4 = True
    else:
        print('KernelGAN-FKP only supports X2 and X4')
        prog.exit(0)

    args.input_dir = '../data/datasets/{}/KernelGANFKP_lr_x{}'.format(args.dataset, 4 if args.X4 else 2)
    args.output_dir = '../data/log_KernelGANFKP/{}_{}_lr_x{}'.format(args.dataset, args.model, 4 if args.X4 else 2)

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
        if args.model == 'KernelGANFKP':
            conf = Config_FKP().parse(create_params(filename, args))
            kernel = train_FKP(conf)
        elif args.model == 'KernelGAN':
            conf = Config().parse(create_params(filename, args))
            kernel = train(conf)

        # nonblind SR
        if args.SR:
            kernel = map2tensor(kernel)
            lr = im2tensor01(read_image(os.path.join(args.input_dir, filename))).unsqueeze(0)

            sr = netG(lr, torch.flip(kernel, [2, 3]), 4 if args.X4 else 2,
                      (10 if args.real else 0) / 255 * torch.ones([1, 1, 1, 1]).cuda())
            plt.imsave(os.path.join(conf.output_dir_path, '%s.png' % conf.img_name), tensor2im01(sr), vmin=0,
                       vmax=1., dpi=1)

    if not conf.verbose and args.SR:
        evaluation_dataset(args.input_dir, conf)

    prog.exit(0)


if __name__ == '__main__':
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

    main()

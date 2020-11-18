import argparse
import torch
import os
import scipy.io as sio
import numpy as np
'''
# ------------------------------------------
# config of KernelGAN-FKP
# ------------------------------------------
'''


class Config_FKP:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.conf = None

        # Paths
        self.parser.add_argument('--img_name', default='', help='image name for saving purposes')
        self.parser.add_argument('--input_image_path', default=os.path.dirname(__file__) + '/training_data/input.png',
                                 help='path to one specific image file')
        self.parser.add_argument('--output_dir_path', default=os.path.dirname(__file__) + '/results',
                                 help='results path')
        self.parser.add_argument('--path_KP', default='KP_x2.pt', help='path for trained kernel prior')

        # Sizes
        self.parser.add_argument('--input_crop_size', type=int, default=64, help='Generators crop size')
        self.parser.add_argument('--scale_factor', type=float, default=0.5, help='The downscaling scale factor')
        self.parser.add_argument('--X4', action='store_true', help='The wanted SR scale factor')

        # Network architecture
        self.parser.add_argument('--G_kernel_size', type=int, default=11, help='The kernel size G is estimating')
        self.parser.add_argument('--D_chan', type=int, default=32, help='# of channels in hidden layer in the D')
        self.parser.add_argument('--D_n_layers', type=int, default=5, help='Discriminators depth')
        self.parser.add_argument('--D_kernel_size', type=int, default=3, help='Discriminators convolution kernels size')

        # Sampling
        self.parser.add_argument('--max_sampling', type=int, default=280000, help='# of sampling')  # 70000

        # Optimization hyper-parameters
        self.parser.add_argument('--g_lr', type=float, default=5e-4, help='initial learning rate for generator')
        self.parser.add_argument('--d_lr', type=float, default=5e-4, help='initial learning rate for discriminator')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='Adam momentum')

        # GPU
        self.parser.add_argument('--gpu_id', type=int, default=0, help='gpu id number')

        # nonblind configuration
        self.parser.add_argument('--SR', action='store_true', help='when activated - nonblind SR is performed')
        self.parser.add_argument('--real', action='store_true', help='if the input is real image')

        # others
        self.parser.add_argument('--verbose', default=False, help='save and output intermediate result')

    def parse(self, args=None):
        """Parse the configuration"""
        self.conf = self.parser.parse_args(args=args)
        self.set_gpu_device()
        self.clean_file_name()
        self.set_output_directory()
        print("Scale Factor: %s \tNonblind: %s" % (('X4' if self.conf.X4 else 'X2'), str(self.conf.SR)))

        if self.conf.real:
            self.conf.kernel_gt = np.ones([11, 11])
        else:
            path = self.conf.input_image_path.replace('lr_x', 'gt_k_x').replace('.png', '.mat')
            self.conf.kernel_gt = sio.loadmat(path)['Kernel']

        return self.conf

    def set_gpu_device(self):
        """Sets the GPU device if one is given"""
        if os.environ.get('CUDA_VISIBLE_DEVICES', '') == '':
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.conf.gpu_id)
            torch.cuda.set_device(0)
        else:
            torch.cuda.set_device(self.conf.gpu_id)

    def clean_file_name(self):
        """Retrieves the clean image file_name for saving purposes"""
        self.conf.img_name = self.conf.input_image_path.split('/')[-1].split('.')[0]

    def set_output_directory(self):
        """Define the output directory name and create the folder"""
        self.conf.output_dir_path += ''
        if self.conf.verbose:
            self.conf.output_dir_path = os.path.join(self.conf.output_dir_path, self.conf.img_name)
            # In case the folder exists - stack 'l's to the folder name
            while os.path.isdir(self.conf.output_dir_path):
                self.conf.output_dir_path += 'l'
            os.makedirs(self.conf.output_dir_path)
        else:
            os.makedirs(self.conf.output_dir_path, exist_ok=True)

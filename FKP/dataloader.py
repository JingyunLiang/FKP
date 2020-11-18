"""
dataloader for FKP
"""
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys

sys.path.append('../')
from data import prepare_dataset


def is_mat_file(filename):
    return any(filename.endswith(extension) for extension in ['.pth'])


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if is_mat_file(fname):
                path = os.path.join(root, fname)
                item = (path, None)
                images.append(item)

    return images


class KernelFolder(data.Dataset):
    """A generic kernel loader"""

    def __init__(self, root, train, kernel_size=11, scale_factor=2, transform=None, target_transform=None,
                 loader=None):
        ''' prepare training and validation sets'''
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor
        self.alpha = 1e-6

        # To normalize the pixels to [0,1], we first clamp the kernel because some values are slightly below zero. Then,
        # we rescale the maximum pixel to be near one by dividing (max_value+0.01), where 0.01 can make sure it won't be
        # larger than 1. This is crucial to remove notable noises in sampling.
        self.normalization = round(prepare_dataset.gen_kernel_fixed(np.array([self.kernel_size, self.kernel_size]),
                                                                    np.array([self.scale_factor, self.scale_factor]),
                                                                    0.175 * self.scale_factor,
                                                                    0.175 * self.scale_factor, 0,
                                                                    0).max(), 5) + 0.01
        root += '_x{}'.format(self.scale_factor)
        if not train:
            if not os.path.exists(root):
                print('generating validation set at {}'.format(root))
                os.makedirs(root, exist_ok=True)

                i = 0
                for sigma1 in np.arange(0.175 * self.scale_factor, min(2.5 * self.scale_factor, 10) + 0.3, 0.3):
                    for sigma2 in np.arange(0.175 * self.scale_factor, min(2.5 * self.scale_factor, 10) + 0.3, 0.3):
                        for theta in np.arange(0, np.pi, 0.2):
                            kernel = prepare_dataset.gen_kernel_fixed(np.array([self.kernel_size, self.kernel_size]),
                                                                      np.array([self.scale_factor, self.scale_factor]),
                                                                      sigma1, sigma2, theta, 0)

                            torch.save(torch.from_numpy(kernel), os.path.join(root, str(i) + '.pth'))
                            i += 1
            else:
                print('Kernel_val_path: {} founded.'.format(root))

            kernels = make_dataset(root, None)

            if len(kernels) == 0:
                raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"))

            self.kernels = kernels

        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (kernel, None)
        """
        if self.train:
            kernel = prepare_dataset.gen_kernel_random(np.array([self.kernel_size, self.kernel_size]),
                                                       np.array([self.scale_factor, self.scale_factor]),
                                                       0.175 * self.scale_factor, min(2.5 * self.scale_factor, 10), 0)
            kernel = torch.from_numpy(kernel)
        else:
            path, target = self.kernels[index]
            kernel = torch.load(path)

        # Normalization
        kernel = torch.clamp(kernel, min=0) / self.normalization

        # Adds noise to pixels to dequantize them, ref MAF. This is crucail to add small numbers to zeros of the kernel.
        # No noise will lead to negative NLL, 720 is an empirical value.
        kernel = kernel + np.random.rand(*kernel.shape) / 720.0

        # Transforms pixel values with logit to be unconstrained by np.log(x / (1.0 - x)), [-13.8,13.8], ref MAF
        kernel = logit(self.alpha + (1 - 2 * self.alpha) * kernel)

        kernel = kernel.to(torch.float32)

        return kernel, torch.zeros(1)

    def __len__(self):
        if self.train:
            return int(5e4)
        else:
            return len(self.kernels)


def logit(x):
    """
    Elementwise logit (inverse logistic sigmoid).
    :param x: numpy array
    :return: numpy array
    """
    return np.log(x / (1.0 - x))

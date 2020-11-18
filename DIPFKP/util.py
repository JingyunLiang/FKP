import os
import time
import torch
import math
import cv2
import numpy as np
from PIL import Image
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from torch.nn import functional as F
from scipy.ndimage import measurements, interpolation
from scipy.interpolate import interp2d
'''
# ------------------------------------------------
# util.py for DIPFKP, etc.
# ------------------------------------------------
'''


def move2cpu(d):
    """Move data from gpu to cpu"""
    return d.detach().cpu().float().numpy()


def tensor2im(im_t):
    """Copy the tensor to the cpu & convert to range [0,255]"""
    im_np = np.clip(np.round((np.transpose(move2cpu(im_t).squeeze(0), (1, 2, 0)) + 1) / 2.0 * 255.0), 0, 255)
    return im_np.astype(np.uint8)


def tensor2im01(im_t):
    """Copy the tensor to the cpu & convert to range [0,255]"""
    im_np = np.clip(np.round((np.transpose(move2cpu(im_t).squeeze(0), (1, 2, 0))) * 255.0), 0, 255)
    return im_np.astype(np.uint8)


def im2tensor(im_np):
    """Copy the image to the gpu & converts to range [-1,1]"""
    im_np = im_np / 255.0 if im_np.dtype == 'uint8' else im_np
    return torch.FloatTensor(np.transpose(im_np, (2, 0, 1)) * 2.0 - 1.0).unsqueeze(0).cuda()


def im2tensor01(im_np):
    """Convert numpy to tensor to the gpu"""
    im_np = im_np / 255.0 if im_np.dtype == 'uint8' else im_np
    return torch.FloatTensor(np.transpose(im_np, (2, 0, 1))).cuda()


def im2tensor01_cpu(im_np):
    """Convert numpy to tensor"""
    im_np = im_np / 255.0 if im_np.dtype == 'uint8' else im_np
    return torch.FloatTensor(np.transpose(im_np, (2, 0, 1)))


def map2tensor(gray_map):
    """Move gray maps to GPU, no normalization is done"""
    return torch.FloatTensor(gray_map).unsqueeze(0).unsqueeze(0).cuda()


def resize_tensor_w_kernel(im_t, k, sf=None):
    """Convolves a tensor with a given bicubic kernel according to scale factor"""
    # Expand dimensions to fit convolution: [out_channels, in_channels, k_height, k_width]
    k = k.expand(im_t.shape[1], im_t.shape[1], k.shape[0], k.shape[1])
    # Calculate padding
    padding = (k.shape[-1] - 1) // 2
    return F.conv2d(im_t, k, stride=round(1 / sf), padding=padding)


def read_image(path):
    """Loads an image"""
    im = Image.open(path).convert('RGB')
    im = np.array(im, dtype=np.uint8)
    return im


def rgb2gray(im):
    """Convert and RGB image to gray-scale"""
    return np.dot(im, [0.299, 0.587, 0.114]) if len(im.shape) == 3 else im


def swap_axis(im):
    """Swap axis of a tensor from a 3 channel tensor to a batch of 3-single channel and vise-versa"""
    return im.transpose(0, 1) if type(im) == torch.Tensor else np.moveaxis(im, 0, 1)


def shave_a2b(a, b):
    """Given a big image or tensor 'a', shave it symmetrically into b's shape"""
    # If dealing with a tensor should shave the 3rd & 4th dimension, o.w. the 1st and 2nd
    is_tensor = (type(a) == torch.Tensor)
    r = 2 if is_tensor else 0
    c = 3 if is_tensor else 1
    # Calculate the shaving of each dimension
    shave_r, shave_c = max(0, a.shape[r] - b.shape[r]), max(0, a.shape[c] - b.shape[c])
    return a[:, :, shave_r // 2:a.shape[r] - shave_r // 2 - shave_r % 2,
           shave_c // 2:a.shape[c] - shave_c // 2 - shave_c % 2] if is_tensor \
        else a[shave_r // 2:a.shape[r] - shave_r // 2 - shave_r % 2,
             shave_c // 2:a.shape[c] - shave_c // 2 - shave_c % 2]


def create_gradient_map(im, window=5, percent=.97):
    """Create a gradient map of the image blurred with a rect of size window and clips extreme values"""
    # Calculate gradients
    gx, gy = np.gradient(rgb2gray(im))
    # Calculate gradient magnitude
    gmag, gx, gy = np.sqrt(gx ** 2 + gy ** 2), np.abs(gx), np.abs(gy)
    # Pad edges to avoid artifacts in the edge of the image
    gx_pad, gy_pad, gmag = pad_edges(gx, int(window)), pad_edges(gy, int(window)), pad_edges(gmag, int(window))
    lm_x, lm_y, lm_gmag = clip_extreme(gx_pad, percent), clip_extreme(gy_pad, percent), clip_extreme(gmag, percent)
    # Sum both gradient maps
    grads_comb = lm_x / lm_x.sum() + lm_y / lm_y.sum() + gmag / gmag.sum()
    # Blur the gradients and normalize to original values
    loss_map = convolve2d(grads_comb, np.ones(shape=(window, window)), 'same') / (window ** 2)
    # Normalizing: sum of map = numel
    return loss_map / np.mean(loss_map)


def create_probability_map(loss_map, crop):
    """Create a vector of probabilities corresponding to the loss map"""
    # Blur the gradients to get the sum of gradients in the crop
    blurred = convolve2d(loss_map, np.ones([crop // 2, crop // 2]), 'same') / ((crop // 2) ** 2)
    # Zero pad s.t. probabilities are NNZ only in valid crop centers
    prob_map = pad_edges(blurred, crop // 2)
    # Normalize to sum to 1
    prob_vec = prob_map.flatten() / prob_map.sum() if prob_map.sum() != 0 else np.ones_like(prob_map.flatten()) / \
                                                                               prob_map.flatten().shape[0]
    return prob_vec


def pad_edges(im, edge):
    """Replace image boundaries with 0 without changing the size"""
    zero_padded = np.zeros_like(im)
    zero_padded[edge:-edge, edge:-edge] = im[edge:-edge, edge:-edge]
    return zero_padded


def clip_extreme(im, percent):
    """Zeroize values below the a threshold and clip all those above"""
    # Sort the image
    im_sorted = np.sort(im.flatten())
    # Choose a pivot index that holds the min value to be clipped
    pivot = int(percent * len(im_sorted))
    v_min = im_sorted[pivot]
    # max value will be the next value in the sorted array. if it is equal to the min, a threshold will be added
    v_max = im_sorted[pivot + 1] if im_sorted[pivot + 1] > v_min else v_min + 10e-6
    # Clip an zeroize all the lower values
    return np.clip(im, v_min, v_max) - v_min


def post_process_k(k, n, sf):
    """Move the kernel to the CPU, eliminate negligible values, and centralize k"""
    k = move2cpu(k)
    # Zeroize negligible values
    significant_k = zeroize_negligible_val(k, n)
    # Force centralization on the kernel
    centralized_k = kernel_shift(significant_k, sf=sf)
    # return shave_a2b(centralized_k, k)
    return centralized_k


def zeroize_negligible_val(k, n):
    """Zeroize values that are negligible w.r.t to values in k"""
    # Sort K's values in order to find the n-th largest
    k_sorted = np.sort(k.flatten())
    # Define the minimum value as the 0.75 * the n-th largest value
    k_n_min = 0.75 * k_sorted[-n - 1]
    # Clip values lower than the minimum value
    filtered_k = np.clip(k - k_n_min, a_min=0, a_max=100)
    # Normalize to sum to 1
    return filtered_k / filtered_k.sum()


def create_penalty_mask(k_size, penalty_scale):
    """Generate a mask of weights penalizing values close to the boundaries"""
    center_size = k_size // 2 + k_size % 2
    mask = create_gaussian(size=k_size, sigma1=k_size, is_tensor=False)
    mask = 1 - mask / np.max(mask)
    margin = (k_size - center_size) // 2 - 1
    mask[margin:-margin, margin:-margin] = 0
    return penalty_scale * mask


def create_gaussian(size, sigma1, sigma2=-1, is_tensor=False):
    """Return a Gaussian"""
    func1 = [np.exp(-z ** 2 / (2 * sigma1 ** 2)) / np.sqrt(2 * np.pi * sigma1 ** 2) for z in
             range(-size // 2 + 1, size // 2 + 1)]
    func2 = func1 if sigma2 == -1 else [np.exp(-z ** 2 / (2 * sigma2 ** 2)) / np.sqrt(2 * np.pi * sigma2 ** 2) for z in
                                        range(-size // 2 + 1, size // 2 + 1)]
    return torch.FloatTensor(np.outer(func1, func2)).cuda() if is_tensor else np.outer(func1, func2)


def nn_interpolation(im, sf):
    """Nearest neighbour interpolation"""
    pil_im = Image.fromarray(im)
    return np.array(pil_im.resize((im.shape[1] * sf, im.shape[0] * sf), Image.NEAREST), dtype=im.dtype)


def analytic_kernel(k):
    """Calculate the X4 kernel from the X2 kernel (for proof see appendix in paper)"""
    k_size = k.shape[0]
    # Calculate the big kernels size
    big_k = np.zeros((3 * k_size - 2, 3 * k_size - 2))
    # Loop over the small kernel to fill the big one
    for r in range(k_size):
        for c in range(k_size):
            big_k[2 * r:2 * r + k_size, 2 * c:2 * c + k_size] += k[r, c] * k
    # Crop the edges of the big kernel to ignore very small values and increase run time of SR
    crop = k_size // 2
    cropped_big_k = big_k[crop:-crop, crop:-crop]
    # Normalize to 1
    return cropped_big_k / cropped_big_k.sum()


def kernel_shift(kernel, sf):
    # There are two reasons for shifting the kernel :
    # 1. Center of mass is not in the center of the kernel which creates ambiguity. There is no possible way to know
    #    the degradation process included shifting so we always assume center of mass is center of the kernel.
    # 2. We further shift kernel center so that top left result pixel corresponds to the middle of the sfXsf first
    #    pixels. Default is for odd size to be in the middle of the first pixel and for even sized kernel to be at the
    #    top left corner of the first pixel. that is why different shift size needed between odd and even size.
    # Given that these two conditions are fulfilled, we are happy and aligned, the way to test it is as follows:
    # The input image, when interpolated (regular bicubic) is exactly aligned with ground truth.

    # First calculate the current center of mass for the kernel
    current_center_of_mass = measurements.center_of_mass(kernel)

    # The idea kernel center
    # for image blurred by filters.correlate
    # wanted_center_of_mass = np.array(kernel.shape) / 2 + 0.5 * (sf - (kernel.shape[0] % 2))
    # for image blurred by F.conv2d. They are the same after kernel.flip([0,1])
    wanted_center_of_mass = (np.array(kernel.shape) - sf) / 2.

    # Define the shift vector for the kernel shifting (x,y)
    shift_vec = wanted_center_of_mass - current_center_of_mass
    # Before applying the shift, we first pad the kernel so that nothing is lost due to the shift
    # (biggest shift among dims + 1 for safety)
    # kernel = np.pad(kernel, np.int(np.ceil(np.max(np.abs(shift_vec)))) + 1, 'constant')

    # Finally shift the kernel and return
    kernel = interpolation.shift(kernel, shift_vec)

    return kernel


def save_final_kernel_png(k, conf, gt_kernel, step=''):
    """saves the final kernel and the analytic kernel to the results folder"""
    os.makedirs(os.path.join(conf.output_dir_path), exist_ok=True)
    savepath_mat = os.path.join(conf.output_dir_path, '%s.mat' % conf.img_name)
    savepath_png = os.path.join(conf.output_dir_path, '%s_kernel.png' % conf.img_name)
    if step != '':
        savepath_mat = savepath_mat.replace('.mat', '_{}.mat'.format(step))
        savepath_png = savepath_png.replace('.png', '_{}.png'.format(step))

    sio.savemat(savepath_mat, {'Kernel': k})
    plot_kernel(gt_kernel, k, savepath_png)


def plot_kernel(gt_k_np, out_k_np, savepath):
    plt.clf()
    f, ax = plt.subplots(1, 2, figsize=(6, 4), squeeze=False)
    im = ax[0, 0].imshow(gt_k_np, vmin=0, vmax=gt_k_np.max())
    plt.colorbar(im, ax=ax[0, 0])
    im = ax[0, 1].imshow(out_k_np, vmin=0, vmax=out_k_np.max())
    plt.colorbar(im, ax=ax[0, 1])
    ax[0, 0].set_title('GT')
    ax[0, 1].set_title('PSNR: {:.2f}'.format(calculate_psnr(gt_k_np, out_k_np, True)))

    plt.savefig(savepath)


def calculate_psnr(img1, img2, is_kernel=False):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse)) if is_kernel else 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def comp_upto_shift(img1, img2, maxshift=5, border=0, min_interval=0.25):
    '''
   compute sum of square differences between two images, after
   finding the best shift between them. need to account for shift
   because the kernel reconstruction is shift invariant- a small
   shift of the image and kernel will not effect the likelihood score.
   Args:
        I1/img1: estimated image
        I2/img2: reference
        ychannel: use ychannel for evaluation, faster and better
        maxshift: assumed maxshift
        boarder: shave boarder to calculate PSNR and SSIM
    '''

    I1 = img1
    I2 = img2

    I2 = I2[border:-border,border:-border]
    I1 = I1[border-maxshift:-border+maxshift,border-maxshift:-border+maxshift]
    N1, N2 = I2.shape[:2]

    gx, gy = np.arange(-maxshift, N2+maxshift, 1.0), np.arange(-maxshift, N1+maxshift, 1.0)

    shifts = np.linspace(-maxshift, maxshift, int(2*maxshift/min_interval+1))
    gx0, gy0 = np.arange(0, N2, 1.0), np.arange(0, N1, 1.0)

    ssdem=np.zeros([len(shifts),len(shifts)])
    for i in range(len(shifts)):
        for j in range(len(shifts)):
            gxn = gx0+shifts[i]
            gvn = gy0+shifts[j]
            if I1.ndim == 2:
                tI1 = interp2d(gx, gy, I1)(gxn, gvn)
            elif I1.ndim == 3:
                tI1 = np.zeros(I2.shape)
                for k in range(I1.shape[-1]):
                    tI1[:,:,k] = interp2d(gx, gy, I1[:,:,k])(gxn, gvn)
            ssdem[i,j]=np.sum((tI1[border:-border, border:-border]-I2[border:-border, border:-border])**2)

    # util.surf(ssdem)
    idxs = np.unravel_index(np.argmin(ssdem), ssdem.shape)
    # print('shifted pixel is {}x{}'.format(shifts[idxs[0]], shifts[idxs[1]]))

    gxn = gx0+shifts[idxs[0]]
    gvn = gy0+shifts[idxs[1]]
    if I1.ndim == 2:
        tI1 = interp2d(gx, gy, I1)(gxn, gvn)
    elif I1.ndim == 3:
        tI1 = np.zeros(I2.shape)
        for k in range(I1.shape[-1]):
            tI1[:,:,k] = interp2d(gx, gy, I1[:,:,k])(gxn, gvn)

    psnr = calculate_psnr(tI1, I2)
    ssim = calculate_ssim(tI1, I2)

    return psnr, ssim

def modcrop(img_in, scale):
    # img_in: Numpy, HWC or HW
    img = np.copy(img_in)
    if img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
    return img


def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def evaluation_dataset(input_dir, conf, used_iter=''):
    ''' Evaluate the model with kernel and image PSNR'''
    print('Calculating PSNR...')
    filesource = os.listdir(os.path.abspath(input_dir))
    filesource.sort()

    im_psnr = 0
    im_ssim = 0
    kernel_psnr = 0
    for filename in filesource:
        # load gt kernel
        if conf.real:
            kernel_gt = np.ones([min(conf.sf * 4 + 3, 21), min(conf.sf * 4 + 3, 21)])
        else:
            path = os.path.join(input_dir, filename).replace('lr_x', 'gt_k_x').replace('.png', '.mat')
            kernel_gt = sio.loadmat(path)['Kernel']

        # load estimated kernel
        path = os.path.join(conf.output_dir_path, filename).replace('.png', '.mat')
        kernel = sio.loadmat(path)['Kernel']

        # calculate psnr
        kernel_psnr += calculate_psnr(kernel_gt, kernel, is_kernel=True)

        # load HR
        path = os.path.join(input_dir.replace(input_dir.split('/')[-1], 'HR'), filename)
        hr = read_image(path)
        hr = modcrop(hr, conf.sf)

        # load SR
        path = os.path.join(conf.output_dir_path, filename)
        sr = read_image(path)

        # calculate psnr
        hr = rgb2ycbcr(hr / 255., only_y=True)
        sr = rgb2ycbcr(sr / 255., only_y=True)
        crop_border = conf.sf
        cropped_hr = hr[crop_border:-crop_border, crop_border:-crop_border]
        cropped_sr = sr[crop_border:-crop_border, crop_border:-crop_border]
        im_psnr += calculate_psnr(cropped_hr * 255, cropped_sr * 255)
        im_ssim += calculate_ssim(cropped_hr * 255, cropped_sr * 255)

        # psnr, ssim = comp_upto_shift(hr * 255, sr*255, maxshift=1, border=conf.sf, min_interval=0.25)
        # im_psnr += psnr
        # im_ssim += ssim


    print('{}_iter{} ({} images), Average Imgae PSNR/SSIM: {:.2f}/{:.4f}, Average Kernel PSNR: {:.2f}'.format(conf.output_dir_path,
                                                                                                  used_iter,
                                                                                                  len(filesource),
                                                                                                  im_psnr / len(
                                                                                                      filesource),
                                                                                                  im_ssim / len(
                                                                                                      filesource),
                                                                                                  kernel_psnr / len(
                                                                                                      filesource)))


########### below are for DIPFKP only
def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    torch.manual_seed(1)
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_()
    else:
        assert False


def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]


def get_noise(input_depth, method, spatial_size, noise_type='u', var=1. / 10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`)
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler.
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)

        fill_noise(net_input, noise_type)
        net_input *= var
    elif method == 'meshgrid':
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1]) / float(spatial_size[1] - 1),
                           np.arange(0, spatial_size[0]) / float(spatial_size[0] - 1))
        meshgrid = np.concatenate([X[None, :], Y[None, :]])
        net_input = np_to_torch(meshgrid)
    else:
        assert False

    return net_input

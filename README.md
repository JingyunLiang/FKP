
# Flow-based Kernel Prior with Application to Blind Super-Resolution (FKP), CVPR2021

This repository is the official PyTorch implementation of Flow-based Kernel Prior with Application to Blind Super-Resolution 
([arxiv](https://arxiv.org/pdf/2103.15977.pdf), [supp](https://github.com/JingyunLiang/FKP/releases)).

:rocket:  :rocket:  :rocket: **News**: 
 - Aug. 18, 2021: See our recent work for [blind SR: Mutual Affine Network for Spatially Variant Kernel Estimation in Blind Image Super-Resolution (MANet), ICCV2021](https://github.com/JingyunLiang/MANet)
 - Aug. 18, 2021: See our recent work for [flow-based generative modelling of image SR: Hierarchical Conditional Flow: A Unified Framework for Image Super-Resolution and Image Rescaling (HCFlow), ICCV2021](https://github.com/JingyunLiang/HCFlow)
 - Aug. 18, 2021: See our recent work for [real-world image SR: Designing a Practical Degradation Model for Deep Blind Image Super-Resolution (BSRGAN), ICCV2021](https://github.com/cszn/BSRGAN)
 
 ---

> Kernel estimation is generally one of the key problems for blind image super-resolution (SR). Recently, Double-DIP proposes to model the kernel via a network architecture prior, while KernelGAN employs the deep linear network and several regularization losses to constrain the kernel space. However, they fail to fully exploit the general SR kernel assumption that anisotropic Gaussian kernels are sufficient for image SR. To address this issue, this paper proposes a normalizing flow-based kernel prior (FKP) for kernel modeling. By learning an invertible mapping between the anisotropic Gaussian kernel distribution and a tractable latent distribution, FKP can be easily used to replace the kernel modeling modules of Double-DIP and KernelGAN. Specifically, FKP optimizes the kernel in the latent space rather than the network parameter space, which allows it to generate reasonable kernel initialization, traverse the learned kernel manifold and improve the optimization stability. Extensive experiments on synthetic and real-world images demonstrate that the proposed FKP can significantly improve the kernel estimation accuracy with less parameters, runtime and memory usage, leading to state-of-the-art blind SR results.
><p align="center">
  > <img height="120" src="./data/illustrations/FKP.png"><img height="120" src="./data/illustrations/DIPFKP.png"><img height="120" src="./data/illustrations/KernelGANFKP.png">
</p>

## Requirements
- Python 3.6, PyTorch >= 1.6 
- Requirements: opencv-python, tqdm
- Platforms: Ubuntu 16.04, cuda-10.0 & cuDNN v-7.5


## Quick Run
To run the code without preparing data, run this command:
```bash
cd DIPFKP
python main.py --SR --sf 4 --dataset Test
```

---

## Data Preparation
To prepare testing data, please organize images as `data/datasets/DIV2K/HR/0801.png`, and run this command:
```bash
cd data
python prepare_dataset.py --model DIPFKP --sf 2 --dataset Set5
python prepare_dataset.py --model KernelGANFKP --sf 2 --dataset DIV2K
```
Commonly used datasets can be downloaded [here](https://github.com/xinntao/BasicSR/blob/master/docs/DatasetPreparation.md#common-image-sr-datasets). Note that KernelGAN/KernelGAN-FKP use analytic X4 kernel based on X2, and do not support X3.

## FKP

To train FKP, run this command:

```bash
cd FKP
python main.py --train --sf 2
```
Pretrained FKP and [USRNet](https://github.com/cszn/KAIR) models are already provided in `data/pretrained_models`.


## DIP-FKP

To test DIP-FKP (no training phase), run this command:

```bash
cd DIPFKP
python main.py --SR --sf 2 --dataset Set5
```


## KernelGAN-FKP

To test KernelGAN-FKP (no training phase), run this command:

```bash
cd KernelGANFKP
python main.py --SR --sf 2 --dataset DIV2K
```

## Results
Please refer to the [paper](https://arxiv.org/pdf/2103.15977.pdf) and [supplementary](https://github.com/JingyunLiang/FKP/releases) for results. Since both DIP-FKP and KernelGAn-FKP are randomly intialized, different runs may get slightly different results. The reported results are averages of 5 runs.



## Citation
```
@article{liang21fkp,
  title={Flow-based Kernel Prior with Application to Blind Super-Resolution},
  author={Liang, Jingyun and Zhang, Kai and Gu, Shuhang and Van Gool, Luc and Timofte, Radu},
  journal={arXiv preprint arXiv:2103.15977},
  year={2021}
}
```


## License & Acknowledgement

This project is released under the Apache 2.0 license. The codes are based on [normalizing_flows](https://github.com/kamenbliznashki/normalizing_flows), [DIP](https://github.com/DmitryUlyanov/deep-image-prior), [KernelGAN](https://github.com/sefibk/KernelGAN) and [USRNet](https://github.com/cszn/KAIR). Please also follow their licenses. Thanks for their great works.



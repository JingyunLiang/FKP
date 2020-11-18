
# Flow-based Kernel Prior with Application to Blind Super-Resolution (FKP)

This repository is the official PyTorch implementation of Flow-based Kernel Prior with Application to Blind Super-Resolution.

[paper](https://arxiv.org/abs/2030.12345) | [supplementary material](https://arxiv.org/abs/2030.12345)

> Kernel estimation is one of the key problems in some blind image super-resolution (SR) methods, among which Double-DIP and KernelGAN have shown great promise. Double-DIP models the kernel via a network architecture prior, while KernelGAN employs deep linear network and several regularization losses to constrain the kernel space. As a result, there is still room for improvement as they fail to fully exploit the general SR kernel prior assumption. To address this issue, this paper proposes a normalizing flow network, dubbed FKP, for kernel prior modeling. By learning an invertible mapping between the complex anisotropic kernel distribution and a tractable Gaussian distribution, FKP can be easily used to replace the kernel modeling modules of both Double-DIP and KernelGAN. Specifically, FKP optimizes the kernel in the Gaussian distribution constrained network input space rather than the network parameter space, which allows it to traverse the learned kernel manifold and search for the best kernel prediction. Extensive experiments on synthetic and real-world images demonstrate that the proposed FKP can enable Double-DIP and KernelGAN to produce more accurate and stable kernel estimation.
><p align="center">
  <img height="150" src="./data/illustrations/FKP.png"><img height="150" src="./data/illustrations/DIPFKP.png"><img height="150" src="./data/illustrations/KernelGANFKP.png">
</p>

## Requirements
- Python 3.6, PyTorch >= 1.6 
- Requirements: opencv-python, tqdm
- Platforms: Ubuntu 16.04, cuda-10.0 & cuDNN v-7.5

## Data Preparation
To prepare testing data, please organize images as `data/datasets/DIV2K/HR/0801.png`, and run this command:
```bash
cd data
python prepare_dataset.py --model DIPFKP --sf 2 --dataset Set5
python prepare_dataset.py --model KernelGANFKP --sf 2 --dataset DIV2K
```
Note that KernelGAN/KernelGAN-FKP uses analytic X4 kernel based on X2, and does not support X3.

## FKP

To train FKP, run this command:

```bash
cd FKP
python main.py --train --sf 2
```
Pretrained FKP and [USRNet]((https://github.com/cszn/KAIR)) are already provided in `data/pretrained_models`.


## DIP-FKP

To test DIP-FKP, run this command:

```bash
cd DIPFKP
python main.py --SR --sf 2 --dataset Set5
```


## KernelGAN-FKP

To test KernelGAN-FKP, run this command:

```bash
cd KernelGANFKP
python main.py --SR --sf 2 --dataset DIV2K
```

## Results
Please refer to the [paper](https://arxiv.org/abs/2030.12345) and the [supplementary material](https://arxiv.org/abs/2030.12345) for results. Since both DIP-FKP and KernelGAn-FKP are randomly intialized, different runs may get slightly different results. The reported results are averages of 5 runs.



## Citation
    @InProceedings{FKP,
        author = {},
        title = {Flow-based Kernel Prior with Application to Blind Super-Resolution},
        booktitle = {},
        month = {},
        year = {}
    }


## License & Acknowledgement

This project is released under the Apache 2.0 license. The codes are based on [normalizing_flows](https://github.com/kamenbliznashki/normalizing_flows), [DIP](https://github.com/DmitryUlyanov/deep-image-prior), [KernelGAN](https://github.com/sefibk/KernelGAN) and [USRNet](https://github.com/cszn/KAIR). Please also follow their licenses. Thanks for their great works.




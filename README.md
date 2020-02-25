# Deep Global Generalized Gaussian Networks
This is an code implementation of CVPR19 paper ([Deep Global Generalized Gaussian Network](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Deep_Global_Generalized_Gaussian_Networks_CVPR_2019_paper.pdf)), created by [Qilong Wang](https://csqlwang.github.io/homepage/) and Li Zhang.

## Introduction
Recently, global covariance pooling (GCP) has shown great advance in improving classification performance of deep convolutional neural networks (CNNs). However, existing deep GCP networks compute covariance pooling of convolutional activations with assumption that activations are sampled from Gaussian distributions, which may not hold in practice and fails to fully characterize the statistics of activations. To handle this issue, this paper proposes a novel deep global generalized Gaussian network (3G-Net), whose core is to estimate a global covariance of generalized Gaussian for modeling the last convolutional activations. Compared with GCP in Gaussian setting, our 3G-Net assumes the distribution of activations follows a generalized Gaussian, which can capture more precise characteristics of activations. However, there exists no analytic solution for parameter estimation of generalized Gaussian, making our 3G-Net challenging. To this end, we first present a novel regularized maximum likelihood estimator for robust estimating covariance of generalized Gaussian, which can be optimized by a modified iterative re-weighted method. Then, to efficiently estimate the covariance of generaized Gaussian under deep CNN architectures, we approximate this re-weighted method by developing an unrolling re-weighted module and a square root covariance layer. In this way, 3GNet can be flexibly trained in an end-to-end manner. The experiments are conducted on large-scale ImageNet-1K and Places365 datasets, and the results demonstrate our 3G-Net outperforms its counterparts while achieving very competitive performance to state-of-the-arts.

## Overview
![Net](https://github.com/csqlwang/3G-Net/blob/master/3G-Net.png)

## Citation

    @inproceedings{wang2019deep,
      title={Deep global generalized Gaussian networks},
      author={Wang, Qilong and Li, Peihua and Hu, Qinghua and Zhu, Pengfei and Zuo, Wangmeng},
      booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      pages={5080-5088},
      year={2019}
    }

## Our environments

- OS: Ubuntu 16.04
- CUDA: 9.0/10.0
- Toolkit: PyTorch 1.0/1.1
- GPU: GTX 2080Ti/TiTan XP

## Installation

1.pytorch installation following [pytorch.org](https://pytorch.org/)

2.`conda install numpy`

3.`conda install torchvision`

3.`conda install pillow==6.1`

## Usage

1. Training on ImageNet: run ` sh ./scripts/ImageNet/train.sh `

2. Testing on ImageNet:  run ` sh ./scripts/ImageNet/val.sh `

3. Training on Places365:  run ` sh ./scripts/Places365/train.sh `

4. Testing on Places365:  run ` sh ./scripts/Places365/val.sh `

*Note that your need to modify  the `dataset path` or `model name` in `train.sh` or `val.sh` for fitting your configurations. Please refer to the file `./scripts/readme.txt` for modifying other parameters.

*If you would like to test the following pre-trained models, you need to modify the loading code in `main.py` as suggested in the file `./scripts/readme.txt`.

## Main Results and Models 

### ImageNet (in 1-Crop testing)
|Models|Top-1 err.(%)|Top-5 err.(%)|BaiduDrive(models)|Extract code|GoogleDrive|
|:----:|:-----------:|:-----------:|:----------------:|:----------:|:---------:|
|ResNet50-3G|21.34|5.70|[ResNet50-3G-ImageNet](https://pan.baidu.com/s/1C8uNk0PJCanDaNwol0gR1Q)|74ot|[ResNet50-3G-ImageNet](https://drive.google.com/open?id=1hN8Q5rlIOQa0YYkcen9jpN9YatPB1j4D)|
|ResNet101-3G|20.40|5.21|[ResNet101-3G-ImageNet](https://pan.baidu.com/s/1J9f39L0FXRlqxORMa0OkJg)|wmzp|[ResNet101-3G-ImageNet](https://drive.google.com/open?id=14vJLFYqlRJyiIHjoG0lOm0RhB1NRF4Xc)|

### Places365 (in 10-Crop testing)
|Models|Top-1 err.(%)|Top-5 err.(%)|BaiduDrive(models)|Extract code|GoogleDrive|
|:----:|:-----------:|:-----------:|:----------------:|:----------:|:---------:|
|ResNet50-3G|43.07|13.44|[ResNet50-3G-Places365](https://pan.baidu.com/s/19da3ZDTZS0AtGP7FjDryvw)|fnkt|[ResNet50-3G-Places365](https://drive.google.com/open?id=1VMVw35h-iW-d4AYH6ecV58_kECDyNzib)|
|ResNet101-3G|42.82|13.00|[ResNet101-3G-Places365](https://pan.baidu.com/s/17N5edFaP1B5YTaWS6ajT0Q)|1dd9|[ResNet101-3G-Places365](https://drive.google.com/open?id=1dOCeQkLBwR3AJSiH8w1qTq9-kT00T7_G)|

## Acknowledgments
We would like to thank the team behind the [iSQRT-COV](https://github.com/jiangtaoxie/fast-MPN-COV) for providing a nice code, and our code is based on it.

## Contact
If you have any questions or suggestions, please feel free to contact us: qlwang@tju.edu.cn; li_zhang@tju.edu.cn.

# Deep global generalized Gaussian networks
This is the link of the paper:[Deep global generalized Gaussian networks](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Deep_Global_Generalized_Gaussian_Networks_CVPR_2019_paper.pdf)

## Citation

    @inproceedings{wang2019deep,
      title={Deep global generalized Gaussian networks},
      author={Wang, Qilong and Li, Peihua and Hu, Qinghua and Zhu, Pengfei and Zuo, Wangmeng},
      booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
      pages={5080--5088},
      year={2019}
    }

## Installation

### Requirements

- Python 3.5+
- PyTorch 1.0+

### Our environments

- OS: Ubuntu 16.04
- CUDA: 9.0/10.0
- Toolkit: PyTorch 1.0/1.1
- GPU: GTX 2080Ti/TiTan XP

### Start Up

#### Train on ImageNet

You can run the `./scripts/ImageNet/train.sh` to train and run `./scripts/ImageNet/val.sh` to evaluate.

#### Train on Places365
You can run the `./scripts/Places365/train.sh` to train and run `./scripts/Places365/val.sh` to evaluate.

## Experiments

### ImageNet
|Model|Top-1 err.(%)|Top-5 err.(%)|BaiduDrive(models)|Extract code|GoogleDrive|
|:---:|:-----------:|:-----------:|:----------------:|:----------:|:---------:|
|ResNet-50+3G-Net|21.34|5.70|[resnet50_3gnet_ImageNet](https://pan.baidu.com/s/1C8uNk0PJCanDaNwol0gR1Q)|74ot|[resnet50_3gnet_ImageNet](https://drive.google.com/open?id=1hN8Q5rlIOQa0YYkcen9jpN9YatPB1j4D)|
|ResNet-101+3G-Net|20.40|5.21|[resnet101_3gnet_ImageNet](https://pan.baidu.com/s/1J9f39L0FXRlqxORMa0OkJg)|wmzp|[resnet101_3gnet_ImageNet](https://drive.google.com/open?id=14vJLFYqlRJyiIHjoG0lOm0RhB1NRF4Xc)|

### Places365
|Model|Top-1 err.(%)|Top-5 err.(%)|BaiduDrive(models)|Extract code|GoogleDrive|
|:---:|:-----------:|:-----------:|:----------------:|:----------:|:---------:|
|ResNet-50+3G-Net|43.07|13.44|[resnet50_3gnet_Places365](https://pan.baidu.com/s/19da3ZDTZS0AtGP7FjDryvw)|fnkt|[resnet50_3gnet_Places365](https://drive.google.com/open?id=1VMVw35h-iW-d4AYH6ecV58_kECDyNzib)|
|ResNet-101+3G-Net|42.82|13.00|[resnet101_3gnet_Places365](https://pan.baidu.com/s/17N5edFaP1B5YTaWS6ajT0Q)|1dd9|[resnet101_3gnet_Places365](https://drive.google.com/open?id=1dOCeQkLBwR3AJSiH8w1qTq9-kT00T7_G)|


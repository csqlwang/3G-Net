#!/bin/sh
now=$(date +"%Y%m%d_%H%M%S")

jobname=ResNet50-3G-Test-ImageNet

log_dir=logs/${jobname}

if [ ! -d $log_dir ]; then
  echo create log $log_dir
  mkdir -p $log_dir
fi

python -u main.py \
       -a resnet50_3g \
       -e \
       -b 256 \
       -j 16 \
       --lr 0.1 \
       --wd 1e-4 \
       --epochs 115 \
       -p 100 \
       --resume ResNet50-3G-ImageNet.pth.tar \
       /home/sdc1/dataset/ILSVRC2012/images | tee ./logs/${jobname}/record-train-${now}.txt \


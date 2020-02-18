#!/bin/sh
now=$(date +"%Y%m%d_%H%M%S")

jobname=Places365-ResNet18

log_dir=logs/${jobname}

if [ ! -d $log_dir ]; then
  echo create log $log_dir
  mkdir -p $log_dir
fi

python -u main.py \
       -a resnet18_365 \
       -b 256 \
       -j 16 \
       --epochs 115 \
       -p 100 \
       --resume checkpoint.pth.tar \
       /home/sdc1/dataset/places365_standard | tee ./logs/${jobname}/record-train-${now}.txt \ 

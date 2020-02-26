#!/bin/sh
now=$(date +"%Y%m%d_%H%M%S")

jobname=ResNet50-3G-Places365

log_dir=logs/${jobname}

if [ ! -d $log_dir ]; then
  echo create log $log_dir
  mkdir -p $log_dir
fi

python -u Train-Places365.py \
       -a resnet50_3g_365 \
       -b 256 \
       -j 16 \
       --lr 0.1 \
       --wd 1e-4 \
       --epochs 55 \
       -p 100 \
       --resume checkpoint.pth.tar \
       /home/sdc1/dataset/places365_standard | tee ./logs/${jobname}/record-train-${now}.txt \ 

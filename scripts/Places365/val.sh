#!/bin/sh
now=$(date +"%Y%m%d_%H%M%S")

jobname=3G-ResNet18-Test-Places365

log_dir=logs/${jobname}

if [ ! -d $log_dir ]; then
  echo create log $log_dir
  mkdir -p $log_dir
fi

python -u main-10-Crop.py \
       -a 3g_resnet18_365 \
       -e \
       -b 64 \
       -j 16 \
       --lr 0.1 \
       --wd 1e-4 \
       --epochs 115 \
       -p 100 \
       --resume model_best.pth.tar \
       /home/sdc1/dataset/places365_standard | tee ./logs/${jobname}/record-train-${now}.txt \


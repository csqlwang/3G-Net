#!/bin/sh
now=$(date +"%Y%m%d_%H%M%S")

jobname=Places365-ResNet101-Test

log_dir=logs/${jobname}

if [ ! -d $log_dir ]; then
  echo create log $log_dir
  mkdir -p $log_dir
fi

python -u main-10-Crop.py \
       -a resnet101_365 \
       -e \
       -b 64 \
       -j 16 \
       --epochs 115 \
       -p 100 \
       --resume model_best_sample.pth.tar \
       /home/sdc1/dataset/places365_standard | tee ./logs/${jobname}/record-train-${now}.txt \


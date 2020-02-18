#!/bin/sh
now=$(date +"%Y%m%d_%H%M%S")

jobname=ResNet101-Test

log_dir=logs/${jobname}

if [ ! -d $log_dir ]; then
  echo create log $log_dir
  mkdir -p $log_dir
fi

python -u main.py \
       -a resnet50 \
       -e \
       -b 256 \
       -j 16 \
       --epochs 115 \
       -p 100 \
       --resume model_best_sample.pth.tar \
       /home/sdc1/dataset/ILSVRC2012/images | tee ./logs/${jobname}/record-train-${now}.txt \


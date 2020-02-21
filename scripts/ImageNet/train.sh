#!/bin/sh
now=$(date +"%Y%m%d_%H%M%S")
python -u main.py \
       -a resnet50_3g \
       -b 256 \
       -j 16 \
       --lr 0.1 \
       --wd 1e-4 \
       --epochs 115 \
       -p 100 \
       --resume checkpoint.pth.tar \
       /home/sdc1/dataset/ILSVRC2012/images | tee ./resnet50-3g-${now}.log \ 

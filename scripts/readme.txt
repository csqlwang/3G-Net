----------------------------------------------parameter modification----------------------------------------------

now=$(date +"%Y%m%d_%H%M%S")

### create the floder to save the log file
jobname=ResNet50-3G-ImageNet

log_dir=logs/${jobname}

if [ ! -d $log_dir ]; then
  echo create log $log_dir
  mkdir -p $log_dir
fi

### train or val 
python -u main.py \
       -a resnet50_3g \			#models
       -b 256 \				#batchsize
       -e \				#evaluate model on validation set
       -j 16 \				#worker
       --lr 0.1 \			#init learning rate
       --wd 1e-4 \			#weight decay
       --epochs 115 \			#total epochs
       -p 100 \				#print frequency
       --resume checkpoint.pth.tar \	#path to latest checkpoint
       /home/sdc1/dataset/ILSVRC2012/images | tee ./logs/${jobname}/record-train-${now}.txt \   #dataset path | output the log file


------------------------------------------------------------------------------------------------------------------

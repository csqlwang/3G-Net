-----------------------parameter modification-----------------------

now=$(date +"%Y%m%d_%H%M%S")

### create the floder to save the log file
jobname=ResNet50-3G-ImageNet

log_dir=logs/${jobname}

if [ ! -d $log_dir ]; then
  echo create log $log_dir
  mkdir -p $log_dir
fi

### train  
python -u main.py \
       -a resnet50_3g \			#models
       -b 256 \				#batchsize
       -j 16 \				#worker
       --lr 0.1 \			#init learning rate
       --wd 1e-4 \			#weight decay
       --epochs 115 \			#total epochs
       -p 100 \				#print frequency
       --resume checkpoint.pth.tar \	#path to latest checkpoint
       /home/sdc1/dataset/ILSVRC2012/images | tee ./logs/${jobname}/record-train-${now}.txt \   #dataset path | output the log file


--------------------------------------------------------------------


----------------modify the code about loading models----------------
### Original code
if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

### Modify(Comment out part of the read-in code)
if args.resume:
        if os.path.isfile(args.resume):
            #print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            #args.start_epoch = checkpoint['epoch']
            #best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            #optimizer.load_state_dict(checkpoint['optimizer'])
            #print("=> loaded checkpoint '{}' (epoch {})"
            #      .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
--------------------------------------------------------------------
# pytorch-multigpu
Multi GPU Training Code for Deep Learning with PyTorch. Train PyramidNet for CIFAR10 classification task. This code is for comparing several ways of multi-GPU training.

# Requirement
- Python 3
- PyTorch 1.0.0+
- TorchVision
- TensorboardX

# Usage
### single gpu
```
cd single_gpu
python train.py 
```

### DataParallel
```
cd data_parallel
python train.py --gpu_devices 0 1 2 3 --batch_size 768
```

### DistributedDataParallel
```
cd dist_parallel
python train.py --gpu_device 0 1 2 3 --batch_size 768
```

# Performance
### single gpu
- batch size: 240
- batch time: 6s
- training time: 22 min 
- gpu util: 99 %
- gpu memory: 10 G

### DataParallel(4 k80)
- batch size: 768
- batch time: 5s
- training time: 5 min 
- gpu util: 99 %
- gpu memory: 10 G * 4

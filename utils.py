from functools import lru_cache
import random
import time
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import defaultdict
import torch.utils.data
from torchvision import datasets, transforms
from scipy.ndimage.interpolation import rotate as scipyrotate
import tqdm
from distill_utils.dataset import Kinetics400, UCF101, HMDB51, miniUCF101, staticHMDB51, staticUCF101, staticUCF50, singleSSv2, singleKinetics400
from networks import MLP, ConvNet, LeNet, AlexNet, AlexNetBN, VGG11, VGG11BN, ResNet18, ResNet18BN_AP, ResNet18BN, VideoConvNetMean, VideoConvNetMLP, VideoConvNetLSTM, VideoConvNetRNN, VideoConvNetGRU, ConvNet3D

def get_dataset(dataset, data_path, num_workers=0,img_size=(112,112),split_num=1,split_id=0,split_mode='mean'):
    if dataset == 'MNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 10
        mean = [0.1307]
        std = [0.3081]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.MNIST(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
        class_names = [str(c) for c in range(num_classes)]

    elif dataset == 'HMDB51':
        # this is a video dataset
        channel = 3
        im_size = img_size 
        num_classes = 51

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]  # use imagenet transform
        
        path = data_path+"/HMDB51"
        assert os.path.exists(path)
        if im_size != (112,112):
            transform = transforms.Compose([transforms.Resize((100,80)),
                                            transforms.RandomCrop(im_size),
                                            #transforms.Resize(im_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=mean, std=std)
                                            ])
        else:
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=mean, std=std)
                                            ])

        dst_train = HMDB51(path, split="train", transform=transform) # no augmentation
        dst_test  = HMDB51(path, split="test", transform=transform)
        print("HMDB51 train: ", len(dst_train), "test: ", len(dst_test))
        class_names = None

    else:
        exit('unknown dataset: %s'%dataset)

    testloader = torch.utils.data.DataLoader(dst_test, batch_size=64, shuffle=False, num_workers=num_workers)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader
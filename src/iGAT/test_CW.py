import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import sys
import os
import argparse
from models.ensemble import Ensemble, Ensemble_max
import numpy as np
from models.resnet import ResNet

norm_std = torch.Tensor([0.2023, 0.1994, 0.2010]).view(1,3,1,1)
norm_mean = torch.Tensor([0.4914, 0.4822, 0.4465]).view(1,3,1,1)
denorm_std = torch.Tensor([1/0.2023, 1/0.1994, 1/0.2010]).view(1,3,1,1)
denorm_mean = torch.Tensor([-0.4914, -0.4822, -0.4465]).view(1,3,1,1)

transform_test = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = torchvision.datasets.CIFAR10(
    root='../data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=500, shuffle=False, num_workers=2)

parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('--model-file', type=str, default='', help="path to the pre-trained model")
parser.add_argument('--leaky-relu', type=int, default=0, help="use leaky_relu or not")
parser.add_argument('--depth', type=int, default=20, help="model depth")
parser.add_argument('--num-classes', type=int, default=10, help="the number of classes")
args = parser.parse_args()

def get_models(as_ensemble=False):
    models = []
    state_dict = torch.load(args.model_file)
    iter_m = state_dict.keys()
    for i in iter_m:
        model = ResNet(depth=args.depth, num_classes=args.num_classes, leaky_relu=args.leaky_relu)
        stc = {}
        for k,v in state_dict[i].items():
          if "normalizer" not in k:
            stc[k.replace('model.','')] = v
        model.load_state_dict(stc)
        model.eval()
        model = model.cuda()
        models.append(model)

    if as_ensemble:
        ensemble = Ensemble(models)
        ensemble.eval()
        return ensemble
    else:
        return models

models = get_models()
ensemble = Ensemble(models)

from attack_cw import *

images = []
labels = []
for n_iter, (image, label) in enumerate(testloader):
    images.append(image)
    labels.append(label)
images = torch.cat(images)
labels = torch.cat(labels)
idxes = np.arange(0,10000,10)
images = images[idxes]
labels = labels[idxes]
cw(ensemble, images, labels, norm_mean, norm_std, args.num_classes, max_iterations=100)


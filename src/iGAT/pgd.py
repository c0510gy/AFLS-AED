import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
# from torch.autograd.gradcheck import zero_gradients
import time
import shutil
import sys
import numpy as np

def pgd(inputs, net, norm_std, norm_mean, denorm_std, denorm_mean, targets=None, step_size=2./255., num_steps=20, epsil=8./255.):

    """
       :param image: Image of size HxWx3
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and
       perturbed image
    """
    input_shape = inputs.shape
    pert_image = inputs.clone().cuda()
    inputs_ori = inputs.clone().cuda()
    scale = 1.0 / norm_std.mean()
    pert_image = pert_image + (torch.rand(inputs.shape).cuda()-0.5) * 2 * epsil * scale
    
    inputs_ori /= denorm_std.view(1,3,1,1).cuda()
    inputs_ori -= denorm_mean.view(1,3,1,1).cuda()
    for ii in range(num_steps):
        pert_image.requires_grad_()
        if pert_image.grad is not None:
            pert_image.grad.data.zero_()
        fs = net.eval()(pert_image)
        
        loss_wrt_label = nn.CrossEntropyLoss()(fs, targets)
        grad = torch.autograd.grad(loss_wrt_label, pert_image, only_inputs=True, create_graph=True, retain_graph=False)[0]
        grad.detach_()

        dr = torch.sign(grad.data)
        pert_image.detach_()
        pert_image += dr * step_size * scale
        
        pert_image /= denorm_std.view(1,3,1,1).cuda()
        pert_image -= denorm_mean.view(1,3,1,1).cuda()
        
        r_tot = pert_image - inputs_ori
        r_tot = torch.clamp(r_tot, min=-epsil, max=epsil)
        pert_image = torch.clamp(inputs_ori + r_tot, 0., 1.)
        pert_image -= norm_mean.view(1,3,1,1).cuda()
        pert_image /= norm_std.view(1,3,1,1).cuda()

    return pert_image.detach()
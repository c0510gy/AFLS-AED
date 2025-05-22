import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
#from torchvision.models import resnet18

# from advertorch.utils import NormalizeByChannelMeanStd

from models.resnet import ResNet, ResNet18
from models.ensemble import Ensemble


###################################
# Models                          #
###################################
def get_models_cifar100(args, train=True, as_ensemble=False, model_file=None, leaky_relu=False):
    models = []
    
    mean = torch.tensor([0.5071, 0.4867, 0.4408], dtype=torch.float32).cuda()
    std = torch.tensor([0.2675, 0.2565, 0.2761], dtype=torch.float32).cuda()
    # normalizer = NormalizeByChannelMeanStd(mean=mean, std=std)
    if model_file:
        print(model_file)
        state_dict = torch.load(model_file)
        if train:
            print('Loading pre-trained models...')
            
    
    iter_m = state_dict.keys() if model_file else range(args.model_num)

    for i in iter_m:
        if args.arch.lower() == 'resnet':
            model = ResNet(depth=args.depth, num_classes=args.num_classes, leaky_relu=leaky_relu)
        else:
            raise ValueError('[{:s}] architecture is not supported yet...')
        # we include input normalization as a part of the model
        model = ModelWrapper(model, mean, std)
        if model_file:
            s_dic = {}
            keys = state_dict[i].keys()
            for k in keys:
              if k != "normalizer.mean" and k != "normalizer.std":
                s_dic[k] = state_dict[i][k]
            # model.load_state_dict(state_dict[i])
            model.load_state_dict(s_dic)
        if train:
            model.train()
        else:
            model.eval()
        model = model.cuda()
        models.append(model)

    if as_ensemble:
        assert not train, 'Must be in eval mode when getting models to form an ensemble'
        ensemble = Ensemble(models)
        ensemble.eval()
        return ensemble
    else:
        return models

def get_models_cifar10(args, train=True, as_ensemble=False, model_file=None, leaky_relu=True):
    models = []
    
    mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32).cuda()
    std = torch.tensor([0.2023, 0.1994, 0.2010], dtype=torch.float32).cuda()
    # normalizer = NormalizeByChannelMeanStd(mean=mean, std=std)
    if model_file:
        print(model_file)
        state_dict = torch.load(model_file)
        if train:
            print('Loading pre-trained models...')
            
    
    iter_m = state_dict.keys() if model_file else range(args.model_num)

    for idx, i in enumerate(iter_m):
        if args.arch.lower() == 'resnet':
            model = ResNet(depth=args.depth, num_classes=args.num_classes, leaky_relu=leaky_relu)
        elif args.arch.lower() == 'resnet18':
            print("ARCH: resnet18")
            model = ResNet18(depth=args.depth, num_classes=args.num_classes, leaky_relu=leaky_relu)
        else:
            raise ValueError('[{:s}] architecture is not supported yet...')
        # we include input normalization as a part of the model
        model = ModelWrapper(model, mean, std)
        if model_file:
            s_dic = {}
            keys = state_dict[i].keys()
            for k in keys:
              if k != "normalizer.mean" and k != "normalizer.std":
                s_dic[k] = state_dict[i][k]
            model.load_state_dict(s_dic)
        if train:
            model.train()
        else:
            model.eval()
        model = model.cuda()
        models.append(model)

    if as_ensemble:
        assert not train, 'Must be in eval mode when getting models to form an ensemble'
        ensemble = Ensemble(models)
        ensemble.eval()
        return ensemble
    else:
        return models
        
def get_ensemble(args, train=False, model_file=None, leaky_relu=False):
    return get_models(args, train, as_ensemble=True, model_file=model_file, leaky_relu=leaky_relu)


class ModelWrapper(nn.Module):
    def __init__(self, model, mean, std):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.mean = mean
        self.std = std

    def forward(self, x):
        x_trans = x - self.mean.view(1,3,1,1).cuda()
        x_trans /= self.std.view(1,3,1,1).cuda()
        return self.model(x_trans)
    
    def get_features(self, x, layer, before_relu=True):
        x_trans = x - self.mean.view(1,3,1,1).cuda()
        x_trans /= self.std.view(1,3,1,1).cuda()
        return self.model.get_features(x_trans, layer, before_relu)


###################################
# data loader                     #
###################################
def get_loaders(args, add_gaussian=False):
    kwargs = {'num_workers': 4,
              'batch_size': args.batch_size,
              'shuffle': True,
              'pin_memory': True}
    if not add_gaussian:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            AddGaussianNoise(0., 0.045) #https://arxiv.org/pdf/1901.09981.pdf
        ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    print("CIFAR100")
    trainset = datasets.CIFAR100(root=args.data_dir, train=True,
                                transform=transform_train,
                                download=True)
    testset = datasets.CIFAR100(root=args.data_dir, train=False,
                                transform=transform_test,
                                download=True)
    trainloader = DataLoader(trainset, **kwargs)
    testloader = DataLoader(testset, num_workers=4, batch_size=100, shuffle=False, pin_memory=True)
    return trainloader, testloader

def get_loaders_cifar10(args, add_gaussian=False):
    kwargs = {'num_workers': 4,
              'batch_size': args.batch_size,
              'shuffle': True,
              'pin_memory': True}
    if not add_gaussian:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            AddGaussianNoise(0., 0.045) #https://arxiv.org/pdf/1901.09981.pdf
        ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    print("CIFAR10")
    trainset = datasets.CIFAR10(root=args.data_dir, train=True,
                                transform=transform_train,
                                download=True)
    testset = datasets.CIFAR10(root=args.data_dir, train=False,
                                transform=transform_test,
                                download=True)
    trainloader = DataLoader(trainset, **kwargs)
    testloader = DataLoader(testset, num_workers=4, batch_size=100, shuffle=False, pin_memory=True)
    return trainloader, testloader
    
def get_loaders_cifar100(args, add_gaussian=False):
    kwargs = {'num_workers': 4,
              'batch_size': args.batch_size,
              'shuffle': True,
              'pin_memory': True}
    if not add_gaussian:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            AddGaussianNoise(0., 0.045) #https://arxiv.org/pdf/1901.09981.pdf
        ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    print("CIFAR100")
    trainset = datasets.CIFAR100(root=args.data_dir, train=True,
                                transform=transform_train,
                                download=True)
    testset = datasets.CIFAR100(root=args.data_dir, train=False,
                                transform=transform_test,
                                download=True)
    trainloader = DataLoader(trainset, **kwargs)
    testloader = DataLoader(testset, num_workers=4, batch_size=100, shuffle=False, pin_memory=True)
    return trainloader, testloader
    
from PIL import Image
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torch.utils.data as data
import os
import numpy as np
class CIFAR10(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        #if not self._check_integrity():
        #    raise RuntimeError('Dataset not found or corrupted.' +
        #                       ' You can use download=True to download it')

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                else:
                    self.train_labels += entry['fine_labels']
                fo.close()

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root
        download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        
def get_testloader(args, train=False, batch_size=100, shuffle=False, subset_idx=None):
    kwargs = {'num_workers': 4,
              'batch_size': batch_size,
              'shuffle': shuffle,
              'pin_memory': True}
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    if subset_idx is not None:
        testset = Subset(datasets.CIFAR100(root=args.data_dir, train=train,
                                transform=transform_test,
                                download=False), subset_idx)
    else:
        testset = datasets.CIFAR100(root=args.data_dir, train=train,
                                transform=transform_test,
                                download=False)
    testloader = DataLoader(testset, **kwargs)
    return testloader

class DistillationLoader_max:
    def __init__(self, seed, target):
        self.seed = iter(seed)
        self.target = iter(target)

    def __len__(self):
        return len(self.seed)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            si, sl, sind = next(self.seed)
            ti, tl, tind = next(self.target)
            return si, sl, sind, ti, tl, tind
        except StopIteration as e:
            raise StopIteration
            
class DistillationLoader:
    def __init__(self, seed, target):
        self.seed = iter(seed)
        self.target = iter(target)

    def __len__(self):
        return len(self.seed)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            si, sl = next(self.seed)
            ti, tl = next(self.target)
            return si, sl, ti, tl
        except StopIteration as e:
            raise StopIteration


###################################
# optimizer and scheduler         #
###################################
def get_optimizers(args, models):
    optimizers = []
    lr = args.lr
    weight_decay = 1e-4
    momentum = 0.9
    for model in models:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                        weight_decay=weight_decay)
        optimizers.append(optimizer)
    return optimizers


def get_schedulers(args, optimizers):
    schedulers = []
    gamma = args.lr_gamma
    intervals = args.sch_intervals
    for optimizer in optimizers:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=intervals, gamma=gamma)
        schedulers.append(scheduler)
    return schedulers


# This is used for training of GAL
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., prob=.5):
        self.std = std
        self.mean = mean
        self.prob = prob
        
    def __call__(self, tensor):
        if random.random() > self.prob:
            return tensor
        else:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

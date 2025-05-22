import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from config import DATASETS

def get_loaders(dataset: str,
                data_dir: str,
                batch_size: int,
                num_workers: int = 4):
    """
    dataset: one of DATASETS keys ("cifar10", "cifar100", "imagenet")
    """
    
    spec = DATASETS[dataset]
    mean, std = spec["mean"], spec["std"]
    cls  = getattr(torchvision.datasets, spec["cls"])

    transform_train = T.Compose([
        torchvision.transforms.Resize((spec["image_size"], spec["image_size"])),
        T.ToTensor(),
    ])
    transform_test = T.Compose([
        torchvision.transforms.Resize((spec["image_size"], spec["image_size"])),
        T.ToTensor(),
    ])
    transform_norm = T.Compose([
        T.Normalize(mean, std),
    ])
    
    if spec["cls"] == 'ImageNet':
        train_ds = cls(root=data_dir, split='train',
                    transform=transform_train)
        test_ds = cls(root=data_dir, split='val',
                    transform=transform_test)
    else:
        train_ds = cls(root=data_dir, train=True, download=True,
                    transform=transform_train)
        test_ds = cls(root=data_dir, train=False, download=True,
                    transform=transform_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers)
    test_loader = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader, transform_norm, (mean, std), spec["image_size"]

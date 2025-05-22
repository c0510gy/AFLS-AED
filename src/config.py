from typing import Dict, List


MODEL_SETTINGS = {
    "num_aug": 4, # how many learned pixel‐masks augmentations to apply
    "recov_dim": 128, # hidden dimensionality of each adapter MLP
    "recov_depth": 4, # how many layers in each adapter MLP
}

BACKBONES: Dict[str, Dict] = {
    # --------------------------------------
    # ResNets
    # --------------------------------------
    "resnet18": {
        "source": "resnet18",
        "model_name": "resnet18",
        "pretrained": True,
        "interm_layers": ['act1', *[f'layer{i+1}[{j}]' for i in range(4) for j in range(2)], 'global_pool'],
        "early_dims": [64, *[64 for _ in range(2)], *[128 for _ in range(2)], *[256 for _ in range(2)], *[512 for _ in range(2)], 64*1*1],
    },
    "resnet110": {
        "source": "resnet110",
        "model_name": "resnet110",
        "pretrained": True,
        "interm_layers": ['relu', *[f'layer{i+1}[{j}]' for i in range(3) for j in range(18)], 'avgpool'],
        "early_dims": [16, *[16 for _ in range(18)], *[32 for _ in range(18)], *[64 for _ in range(18)], 64*1*1],
    },

    # --------------------------------------
    # DenseNets
    # --------------------------------------
    "densenet121": {
        "source": "densenet121",
        "model_name": "densenet121",
        "pretrained": True,
        "interm_layers": ['features.pool0', \
                          *sum([[f'features.denseblock1.denselayer{i+1}.conv1', f'features.denseblock1.denselayer{i+1}.conv2'] for i in range(6)], start=[]), \
                          'features.transition1.pool',
                          *sum([[f'features.denseblock2.denselayer{i+1}.conv1', f'features.denseblock2.denselayer{i+1}.conv2'] for i in range(12)], start=[]), \
                          'features.transition2.pool',
                          *sum([[f'features.denseblock3.denselayer{i+1}.conv1', f'features.denseblock3.denselayer{i+1}.conv2'] for i in range(24)], start=[]), \
                          'features.transition3.pool',
                          *sum([[f'features.denseblock4.denselayer{i+1}.conv1', f'features.denseblock4.denselayer{i+1}.conv2'] for i in range(16)], start=[]), \
                          'features.norm5'],
        "early_dims": [64, \
                            *sum([[128, 32] for _ in range(6)], start=[]), \
                            128,
                            *sum([[128, 32] for _ in range(12)], start=[]), \
                            256,
                            *sum([[128, 32] for _ in range(24)], start=[]), \
                            512,
                            *sum([[128, 32] for _ in range(16)], start=[]), \
                            1024
                        ],
    },

    # --------------------------------------
    # ShuffleNetV2
    # --------------------------------------
    "shufflenetv2_x1_0": {
        "source": "torch_hub",
        "repo": "chenyaofo/pytorch-cifar-models",
        "model_name": "cifar100_shufflenetv2_x1_0",
        "pretrained": True,
        "interm_layers": ['conv1', *[f'stage2[{i}]' for i in range(4)], *[f'stage3[{i}]' for i in range(8)], *[f'stage4[{i}]' for i in range(4)], 'conv5'],
        "early_dims": [24, 116, 116, 116, 116, 232, 232, 232, 232, 232, 232, 232, 232, 464, 464, 464, 464, 1024],
    },

    # --------------------------------------
    # MobileNetV2
    # --------------------------------------
    "mobilenetv2_x0_5": {
        "source": "torch_hub",
        "repo": "chenyaofo/pytorch-cifar-models",
        "model_name": "cifar100_mobilenetv2_x0_5",
        "pretrained": True,
        "interm_layers": [f'features[{i}]' for i in range(19)],
        "early_dims": [16, 8, 16, 16, 16, 16, 16, 32, 32, 32, 32, 48, 48, 48, 80, 80, 80, 160, 1280],
    },

    # --------------------------------------
    # RepVGG‐A0
    # --------------------------------------
    "repvgg_a0": {
        "source": "torch_hub",
        "repo": "chenyaofo/pytorch-cifar-models",
        "model_name": "cifar100_repvgg_a0",
        "pretrained": True,
        "interm_layers": ['stage0', *[f'stage1[{i}]' for i in range(2)], *[f'stage2[{i}]' for i in range(4)], *[f'stage3[{i}]' for i in range(14)], *[f'stage4[{i}]' for i in range(1)], 'gap'],
        "early_dims": [48, 48, 48, 96, 96, 96, 96, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 1280, 1280],
    },
}

DATASETS: Dict[str, Dict] = {
    "cifar10": {"cls": "CIFAR10",  "num_classes": 10,
                 "mean": [0.49139968, 0.48215841, 0.44653091],
                 "std": [0.24703223, 0.24348513, 0.26158784],
                 "image_size": 32},
    "cifar100": {"cls": "CIFAR100", "num_classes": 100,
                 "mean": [0.5071, 0.4867, 0.4408],
                 "std": [0.2675, 0.2565, 0.2761],
                 "image_size": 32},
    "imagenet": {"cls": "ImageNet", "num_classes": 1000,
                 "mean": [0.485, 0.456, 0.406],
                 "std": [0.229, 0.224, 0.225],
                 "image_size": 256},
}

ATTACKS = ["FGSM", "PGD", "CW", "AutoAttack", "Square"]

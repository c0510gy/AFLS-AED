import torch
import torch.nn as nn
import torchvision
import timm
import resnet
from config import BACKBONES

def load_backbone(name: str,
                  num_classes: int,
                  device: torch.device):
    
    spec = BACKBONES[name]
    src  = spec["source"]

    if src == "timm":
        model = timm.create_model(
            spec["model_name"],
            pretrained=spec["pretrained"],
            num_classes=num_classes
        )
    elif src == "torch_hub":
        model = torch.hub.load(
            spec["repo"],
            spec["model_name"],
            pretrained=spec["pretrained"]
        )
    elif src == "resnet110":
        model = resnet.resnet110(pretrained=False)
        if spec["pretrained"]:
            model.load_state_dict(torch.load('./backbone_trained/ResNet110_CIFAR10.pt', weights_only=True))
    elif src == "densenet121":
        model = torchvision.models.densenet121(pretrained=spec["pretrained"])
    elif src == "resnet18":
        model = timm.create_model("hf_hub:edadaltocg/resnet18_cifar100", num_classes=100, pretrained=False)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.maxpool = nn.Identity()
        if spec["pretrained"]:
            model.load_state_dict(
                    torch.hub.load_state_dict_from_url(
                        "https://huggingface.co/edadaltocg/resnet18_cifar100/resolve/main/pytorch_model.bin",
                        map_location=device,
                        )
                    )
    else:
        raise ValueError(f"Unknown source {src}")

    model.to(device).eval()
    return model, spec["interm_layers"], spec["early_dims"]

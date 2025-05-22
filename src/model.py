import torch
import torch.nn as nn
import torch.nn.functional as F
from config import MODEL_SETTINGS


class Tester(nn.Module):
    def __init__(self,
                 early_dims: list,
                 image_size: int):
        super().__init__()
        
        d_hidden = MODEL_SETTINGS["recov_dim"]
        depth = MODEL_SETTINGS["recov_depth"]

        # build one adapter per early feature
        self.adapters = nn.ModuleList()
        for out_dim in early_dims[:-1]:
            layers = [ nn.Linear(early_dims[-1], d_hidden),
                       nn.BatchNorm1d(d_hidden),
                       nn.ReLU() ]
            for _ in range(depth - 2):
                layers += [ nn.Linear(d_hidden, d_hidden),
                            nn.BatchNorm1d(d_hidden),
                            nn.ReLU() ]
            layers += [ nn.Linear(d_hidden, out_dim) ]
            self.adapters.append(nn.Sequential(*layers))

        # learned multiplicative augmentation masks
        self.aug_masks = nn.Parameter(
            torch.ones(MODEL_SETTINGS["num_aug"], 3, image_size, image_size)
        )

    def forward(self, early_feats: list):
        deepest = early_feats[-1]
        guesses = [ adapter(deepest) for adapter in self.adapters ]
        return guesses


class ModelWrapper(nn.Module):
    def __init__(self, backbone: nn.Module, hook_layers: list, early_dims: list, num_classes: int, image_size: int):
        super().__init__()
        
        self.backbone = backbone
        self.hook_layers = hook_layers
        self.early_dims  = early_dims
        self.num_classes = num_classes
        self.tester_model = Tester(early_dims, image_size)
        
        self._activations = {}
        
        def _make_hook(name):
            def hook(_, __, out):
                # out: Tensor[B, C, H, W]
                self._activations[name] = out
            return hook

        # register forward‐hooks to capture the chosen layers
        for name in hook_layers:
            module = eval(f"self.backbone.{name}")
            module.register_forward_hook(_make_hook(name))

    def _num_classes(self):
        if hasattr(self.backbone, "fc"):
            return self.backbone.fc.out_features
        else:
            return self.backbone.classifier[-1].out_features


    def forward(self, x, unnorm=None, transform_norm=None, during_train=False):
        """
        x               : normalized input tensor (B, 3, image_size, image_size)
        unnorm          : the raw input if you want to apply masks to it
        during_train    : whether to leave graph open for recover-module grads
        """
        
        B = x.size(0)

        logits = self.backbone(x) # (B, num_classes)
        early_feats = []
        for dim, name in zip(self.early_dims, self.hook_layers):
            feat = self._activations[name].detach().float() # (B, C, H, W)
            feat = feat.view(B, dim, -1).mean(-1) # (B, dim)
            early_feats.append(feat)

        aug_early_feats = [] # [ (B, dim1), (B, dim2), ... ]
        aug_logits = [] # [ (B, num_classes), ... ]
        for mask in self.tester_model.aug_masks:
            x_aug = unnorm * mask.unsqueeze(0) # (B, 3, H, W)
            logit_aug = self.backbone(x_aug if transform_norm is None else transform_norm(x_aug)) # (B, num_classes)
            aug_logits.append(logit_aug)

            feats_aug = []
            for dim, name in zip(self.early_dims, self.hook_layers):
                feat = self._activations[name].float() #.detach().float()
                feat = feat.view(B, dim, -1).mean(-1)
                feats_aug.append(feat)
            aug_early_feats.append(feats_aug)

        if during_train:
            easy_guesses = self.tester_model(early_feats)
        else:
            with torch.no_grad():
                easy_guesses = self.tester_model(early_feats)
        
        recover_error = torch.stack([
            torch.sum(
                (early_feats[l].detach() - easy_guesses[l])**2
            , dim=1) for l in range(len(easy_guesses))
        ]).permute(1, 0) # (B, L - 1)
        recoverability = (torch.log(torch.tensor(len(easy_guesses))) + \
                          torch.softmax(recover_error, dim=-1) * torch.log(torch.softmax(recover_error, dim=-1) + 1e-8)) * \
                          torch.log(recover_error.mean(-1) + 1e-8).reshape(-1, 1) # (B, L - 1)

        inter_mses = []
        for feats_aug in aug_early_feats:
            mses = torch.stack([
                torch.mean(
                    (feats_aug[l] - early_feats[l].detach())**2
                , dim=1) for l in range(len(feats_aug))
            ]).permute(1, 0)
            inter_mses.append(mses)
        inter_mses = torch.stack(inter_mses, dim=1) # (B, num_aug, L)

        orig_probs = torch.softmax(logits.detach(), dim=-1) # (B, C)
        orig_onehots = F.one_hot(torch.max(logits.detach(), dim=-1).indices, num_classes=self.num_classes)
        entropies = torch.sum(-orig_probs * torch.log(orig_probs), dim=-1) # (B, )
        inter_logits = []
        for logit_aug in aug_logits:
            p_aug = torch.softmax(logit_aug, dim=-1)
            inter_logits.append( torch.mean((p_aug - orig_onehots)**2, dim=-1) * entropies )
        inter_logits = torch.stack(inter_logits, dim=1)  # (B, num_aug)
        
        inter_logits = (torch.log(inter_logits.unsqueeze(2) + 1e-8) - torch.log(inter_mses + 1e-8))
        
        return (
            logits, # (B, num_classes)
            early_feats, # [ (B, dim), ... ]
            easy_guesses, # [ (B, dim), ... ]
            aug_logits, # [ (B, num_classes), ... ]
            recoverability, # (B, 1)
            inter_logits, # (B, num_aug)
        )

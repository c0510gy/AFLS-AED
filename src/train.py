import argparse
import torch
import torch.optim as optim
import tqdm

from config import BACKBONES, DATASETS, MODEL_SETTINGS, ATTACKS
from data import get_loaders
from backbone import load_backbone
from model import ModelWrapper
from attacks import gen_fgsm, gen_pgd, gen_cw, gen_autoattack, gen_square
from utils import compute_auc
from functools import partial
from sklearn.preprocessing import QuantileTransformer
import matplotlib.pyplot as plt
import numpy as np


def train_epoch(model, loader, opt, transform_norm, device, train_break_point=-1):
    
    model.train()
    total, correct = 0, 0
    steps = 0.
    running_loss = 0.
    
    for x, y in tqdm.tqdm(loader, desc="Train"):
        
        x, y = x.to(device), y.to(device)
        
        model.train()
        model.backbone.eval()
        
        opt.zero_grad()
        
        logits, early_feats, easy_guesses, aug_logits, recoverability, inter_logits = \
            model(transform_norm(x), unnorm=x, during_train=True)
        
        recov_loss = torch.mean(sum([
            torch.mean((early_feats[i].detach() - easy_guesses[i])**2, dim=1)
            for i in range(len(easy_guesses))
        ]))
        loss = recov_loss + inter_logits.mean()
        
        loss.backward()
        running_loss += loss.item()
        opt.step()

        preds = logits.argmax(dim=1)
        total += y.size(0)
        correct += (preds == y).sum().item()
        
        steps += 1
        
        if train_break_point > 0 and steps > train_break_point: break

    return correct / total, running_loss / steps


def validate(model, loader, transform_norm, device):
    
    model.eval()
    total, correct = 0, 0
    steps = 0.
    running_loss = 0.
    
    with torch.no_grad():
        for x, y in tqdm.tqdm(loader, desc="Validate"):
            
            x, y = x.to(device), y.to(device)
            
            logits, early_feats, easy_guesses, aug_logits, recoverability, inter_logits = \
                model(transform_norm(x), unnorm=x, during_train=False)
            
            recov_loss = torch.mean(sum([
                torch.mean((early_feats[i].detach() - easy_guesses[i])**2, dim=1)
                for i in range(len(easy_guesses))
            ]))
            loss = recov_loss + inter_logits.mean()
            
            running_loss += loss.item()
            
            preds = logits.argmax(dim=1)
            total += y.size(0)
            correct += (preds == y).sum().item()
            
            steps += 1
            
    return correct / total, running_loss / steps


def validate_detection_and_robust(model, loader, attack_fn, transform_norm, mean, std, device, k_rec=0, k_logit=0):
    
    model.eval()
    bs = loader.batch_size

    # Generate adversarial examples
    adv_samples, adv_targets = attack_fn(model.backbone, loader, mean, std, device=device)
    adv_dataset = torch.utils.data.TensorDataset(adv_samples, adv_targets)
    adv_loader = torch.utils.data.DataLoader(
        adv_dataset, batch_size=bs, shuffle=False, num_workers=4
    )

    clean_RT_scores = []
    adv_RT_scores = []
    clean_LT_scores = []
    adv_LT_scores = []
    correct_robust = 0
    total = 0

    it_clean = iter(loader)
    it_adv = iter(adv_loader)

    with torch.no_grad():
        for _ in range(len(adv_loader)):
            x, y = next(it_clean)
            adv_x, adv_y = next(it_adv)

            x, y = x.to(device), y.to(device)
            adv_x, adv_y = adv_x.to(device), adv_y.to(device)

            logits_adv, early_feats_adv, easy_guesses_adv, aug_logits_adv, recoverability_adv, inter_logits_adv = \
                model(transform_norm(adv_x), unnorm=adv_x, transform_norm=transform_norm, during_train=False)
            preds_adv = logits_adv.argmax(dim=1)
            correct_robust += (preds_adv == adv_y).sum().item()
            total += adv_y.size(0)

            logits_clean, early_feats_clean, easy_guesses_clean, aug_logits_clean, recoverability_clean, inter_logits_clean = \
                model(transform_norm(x), unnorm=x, during_train=False)
            preds_clean = logits_clean.argmax(dim=1)
            
            adversals = ((y == preds_clean) & (preds_clean != preds_adv))
            normals = ((y == preds_clean))
            
            recoverability_clean = recoverability_clean[:,k_rec:].sum(dim=-1)
            recoverability_adv = recoverability_adv[:,k_rec:].sum(dim=-1)

            inter_logits_clean = inter_logits_clean[:,:,k_logit:].mean(-1).mean(-1)
            inter_logits_adv = inter_logits_adv[:,:,k_logit:].mean(-1).mean(-1)
        
            clean_RT_scores.extend(recoverability_clean[normals].cpu().tolist())
            clean_LT_scores.extend(inter_logits_clean[normals].cpu().tolist())
            
            adv_RT_scores.extend(recoverability_adv[adversals].cpu().tolist())
            adv_LT_scores.extend(inter_logits_adv[adversals].cpu().tolist())

    clean_RT_scores = torch.tensor(clean_RT_scores)
    clean_LT_scores = torch.tensor(clean_LT_scores)
    
    adv_RT_scores = torch.tensor(adv_RT_scores)
    adv_LT_scores = torch.tensor(adv_LT_scores)
    
    qt = QuantileTransformer(output_distribution='normal').fit(clean_RT_scores.reshape(-1, 1).numpy())
    clean_RT_scores = torch.from_numpy(qt.transform(clean_RT_scores.reshape(-1, 1).numpy())).reshape(-1)
    adv_RT_scores = torch.from_numpy(qt.transform(adv_RT_scores.reshape(-1, 1).numpy())).reshape(-1)

    qt = QuantileTransformer(output_distribution='normal').fit(clean_LT_scores.reshape(-1, 1).numpy())
    clean_LT_scores = torch.from_numpy(qt.transform(clean_LT_scores.reshape(-1, 1).numpy())).reshape(-1)
    adv_LT_scores = torch.from_numpy(qt.transform(adv_LT_scores.reshape(-1, 1).numpy())).reshape(-1)
    
    robust_acc = correct_robust / total
    detect_RT_auc = compute_auc(clean_RT_scores**2, adv_RT_scores**2)
    detect_LT_auc = compute_auc(clean_LT_scores**2, adv_LT_scores**2)
    
    return robust_acc, detect_RT_auc, detect_LT_auc, (clean_RT_scores, adv_RT_scores), (clean_LT_scores, adv_LT_scores)


ATTACK_FUN = {
    "FGSM": gen_fgsm,
    "PGD": gen_pgd,
    "CW": gen_cw,
    "AutoAttack": gen_autoattack,
    "Square": gen_square,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS.keys(), required=True)
    parser.add_argument("--arch", choices=BACKBONES.keys(), required=True)
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-2)
    parser.add_argument("--train_break", type=bool, default=False)
    parser.add_argument("--num_aug", type=int, default=None)
    parser.add_argument("--recov_dim", type=int, default=None)
    parser.add_argument("--recov_depth", type=int, default=None)
    parser.add_argument("--k_rec", type=int, default=0)
    parser.add_argument("--k_logit", type=int, default=0)
    parser.add_argument(
        "--attacks", nargs="+", choices=ATTACKS,
        default=["FGSM", "PGD"] # "CW" , "AutoAttack", "Square"
    )
    args = parser.parse_args()

    # override MODEL_SETTINGS
    if args.num_aug is not None: MODEL_SETTINGS["num_aug"] = args.num_aug
    if args.recov_dim is not None: MODEL_SETTINGS["recov_dim"] = args.recov_dim
    if args.recov_depth is not None: MODEL_SETTINGS["recov_depth"] = args.recov_depth

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # data loaders + normalization transform
    train_loader, test_loader, transform_norm, (mean, std), image_size = get_loaders(
        args.dataset, args.data_dir, args.batch_size
    )

    # build model
    backbone, hooks, early_dims = load_backbone(
        args.arch,
        DATASETS[args.dataset]["num_classes"],
        device
    )
    model = ModelWrapper(backbone, hooks, early_dims, DATASETS[args.dataset]["num_classes"], image_size).to(device)

    opt = optim.AdamW(
        [
            {
                "params": [
                    p for n, p in model.tester_model.named_parameters()
                    if "adapters" in n
                ],
                "lr": args.lr,
            },
            {
                "params": [model.tester_model.aug_masks],
                "lr": args.lr * 0.01, # Slightly change augmentation matrices
            },
        ],
        weight_decay=args.wd,
    )
    
    max_RT_scores_by_attack, max_LT_scores_by_attack = dict([(atk, None) for atk in args.attacks]), dict([(atk, None) for atk in args.attacks])

    # training
    for epoch in range(1, args.epochs + 1):
        tr_acc, tr_loss = train_epoch(model, train_loader, opt, transform_norm, device, train_break_point=100 if args.train_break else -1)
        te_acc, te_loss = validate(model, test_loader, transform_norm, device)
        print(f"Epoch {epoch:03d}  train={tr_acc:.4f}  test={te_acc:.4f}  tr_loss={tr_loss:.4f}  te_loss={te_loss:.4f}")

        # robust & detection evaluation
        for atk in args.attacks:
            atk_fn = ATTACK_FUN[atk]
            atk_fn = partial(atk_fn, filename=f'attack_samples_{args.dataset}_{args.arch}_{atk}.pt')
            robust_acc, detect_RT_auc, detect_LT_auc, (clean_RT_scores, adv_RT_scores), (clean_LT_scores, adv_LT_scores) = validate_detection_and_robust(
                model, test_loader, atk_fn, transform_norm, mean, std, device, k_rec=args.k_rec, k_logit=args.k_logit
            )
            
            if max_RT_scores_by_attack[atk] is None or max_RT_scores_by_attack[atk][0] < detect_RT_auc:
                max_RT_scores_by_attack[atk] = (detect_RT_auc, clean_RT_scores.clone(), adv_RT_scores.clone())
            if max_LT_scores_by_attack[atk] is None or max_LT_scores_by_attack[atk][0] < detect_LT_auc:
                max_LT_scores_by_attack[atk] = (detect_LT_auc, clean_LT_scores.clone(), adv_LT_scores.clone())
            
            max_RT_scores = max_RT_scores_by_attack[atk]
            max_LT_scores = max_LT_scores_by_attack[atk]
            
            clean_RLT_scores = (max_RT_scores[1] - torch.mean(max_RT_scores[1])).abs()**2 / torch.std(max_RT_scores[1]) + (max_LT_scores[1] - torch.mean(max_LT_scores[1])).abs()**2 / torch.std(max_LT_scores[1])
            adv_RLT_scores = (max_RT_scores[2] - torch.mean(max_RT_scores[1])).abs()**2 / torch.std(max_RT_scores[1]) + (max_LT_scores[2] - torch.mean(max_LT_scores[1])).abs()**2 / torch.std(max_LT_scores[1])
            detect_RLT_auc = compute_auc(clean_RLT_scores, adv_RLT_scores)
            
            print(f"{atk:10s}  robust_acc={robust_acc:.4f}  detect_RT_auc={detect_RT_auc:.4f}  detect_LT_auc={detect_LT_auc:.4f}  detect_RLT_auc={detect_RLT_auc:.4f}")

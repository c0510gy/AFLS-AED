import torch
import tqdm
import foolbox as fb
import os
from iGAT.autoattack.autoattack import AutoAttack

def gen_fgsm(model, loader, mean, std, eps=0.05, filename=None, device="cpu"):
    if filename is not None and os.path.exists(filename):
        advdataset = torch.load(filename)
        advs = advdataset['adv_samples']
        tgts = advdataset['adv_targets']
        return advs, tgts
    
    fmodel = fb.PyTorchModel(model, bounds=(0,1),
        preprocessing=dict(mean=mean, std=std, axis=-3),
        device=device)
    atk = fb.attacks.FGSM()
    advs, tgts = [], []
    for x,y in tqdm.tqdm(loader, desc="FGSM"):
        raw, adv, succ = atk(fmodel, x.to(device), y.to(device), epsilons=[eps])
        advs.append(adv[0].cpu())
        tgts.append(y)
    advs = torch.cat(advs, 0)
    tgts = torch.cat(tgts, 0)
    if filename is not None:
        torch.save({
            'adv_samples': advs.cpu(),
            'adv_targets': tgts.cpu(),
        }, filename)
    return advs, tgts

def gen_pgd(model, loader, mean, std, eps=0.02, filename=None, device="cpu"):
    if filename is not None and os.path.exists(filename):
        advdataset = torch.load(filename)
        advs = advdataset['adv_samples']
        tgts = advdataset['adv_targets']
        return advs, tgts
    
    fmodel = fb.PyTorchModel(model, bounds=(0,1),
        preprocessing=dict(mean=mean, std=std, axis=-3),
        device=device)
    atk = fb.attacks.LinfPGD(abs_stepsize=0.002, steps=50)
    advs, tgts = [], []
    for x,y in tqdm.tqdm(loader, desc="PGD"):
        raw, adv, succ = atk(fmodel, x.to(device), y.to(device), epsilons=[eps])
        advs.append(adv[0].cpu())
        tgts.append(y)
    advs = torch.cat(advs, 0)
    tgts = torch.cat(tgts, 0)
    if filename is not None:
        torch.save({
            'adv_samples': advs.cpu(),
            'adv_targets': tgts.cpu(),
        }, filename)
    return advs, tgts

def gen_autoattack(model, loader, mean, std, eps=8/255, attacks_to_run=None, filename=None, desc='AutoAttack', device="cpu"):
    if filename is not None and os.path.exists(filename):
        advdataset = torch.load(filename)
        advs = advdataset['adv_samples']
        tgts = advdataset['adv_targets']
        return advs, tgts
    
    mean_norm = torch.tensor(mean).view(1,3,1,1).to(device)
    std_norm = torch.tensor(std).view(1,3,1,1).to(device)
    
    attack = AutoAttack(model, norm='Linf', eps=eps, version='standard', mean_norm=mean_norm, std_norm=std_norm, n_itr=None)
    if attacks_to_run is not None:
        attack.attacks_to_run = attacks_to_run
    adv_samples = None
    adv_targets = []
    
    for samples, targets in tqdm.tqdm(loader, desc=desc):
        # Warning: it seems that the output is a probability distribution, please be sure that the logits are used! See flags_doc.md for details.
        advs, robust_accuracy = attack.run_standard_evaluation(samples.clone().to(device), targets.clone().long().to(device), bs=len(samples))

        if adv_samples is None:
            adv_samples = advs.cpu()
        else:
            adv_samples = torch.cat((adv_samples, advs.cpu()), 0)
        
        adv_targets.extend(targets.cpu().tolist())

    adv_targets = torch.tensor(adv_targets)
    
    if filename is not None:
        torch.save({
            'adv_samples': adv_samples.cpu(),
            'adv_targets': adv_targets.cpu(),
        }, filename)
    
    return adv_samples, adv_targets

def gen_cw(model, loader, mean, std, eps=8/255, filename=None, device="cpu"):
    advdataset = torch.load(filename)
    advs = advdataset['adv_samples']
    tgts = advdataset['adv_targets']
    return advs, tgts

def gen_square(model, loader, mean, std, eps=0.05, filename=None, device="cpu"):
    return gen_autoattack(model, loader, mean, std, eps=eps, filename=filename, attacks_to_run=['square'], desc='Square', device=device)

import os, sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
sys.path.append(os.getcwd())
import json, argparse, random
from tqdm import tqdm

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
#from torch.utils.tensorboard import SummaryWriter

import numpy as np
import arguments, utils
from models.ensemble import Ensemble, Ensemble_max
from distillation import Linf_PGD, Linf_distillation


class DVERGE_Trainer():
    def __init__(self, models, optimizers, schedulers,
                 trainloader, testloader, save_root=None, **kwargs):
        self.models = models
        self.epochs = kwargs['epochs']
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.trainloader = trainloader
        self.testloader = testloader
        self.reg_type = kwargs['reg_type']
        self.combine_type = kwargs['combine_type']
        self.alpha = kwargs['igat_alpha']
        self.beta = kwargs['igat_beta']
        self.num_classes = kwargs['num_classes']
        self.save_root = save_root
        self.lr = kwargs['lr']
        self.criterion = nn.CrossEntropyLoss()
        self.criterion_sum = nn.CrossEntropyLoss(reduction="sum")
        self.best_acc = 0.

        # distillation configs
        self.distill_fixed_layer = kwargs['distill_fixed_layer']
        self.distill_cfg = {'eps': kwargs['distill_eps'], 
                           'alpha': kwargs['distill_alpha'],
                           'steps': kwargs['distill_steps'],
                           'layer': kwargs['distill_layer'],
                           'rand_start': kwargs['distill_rand_start'],
                           'before_relu': True,
                           'momentum': kwargs['distill_momentum']
                          }
        
        # diversity training configs
        self.plus_adv = kwargs['plus_adv']
        self.coeff = kwargs['dverge_coeff']
        if self.plus_adv:
            self.attack_cfg = {'eps': kwargs['eps'], 
                               'alpha': kwargs['alpha'],
                               'steps': kwargs['steps'],
                               'is_targeted': False,
                               'rand_start': True
                              }
        self.depth = kwargs['depth']
    
    def get_epoch_iterator(self):
        iterator = tqdm(list(range(1,self.epochs+1)), total=self.epochs, desc='Epoch',
                        leave=True, position=1)
        return iterator
    
    def get_batch_iterator(self):
        loader = utils.DistillationLoader(self.trainloader, self.trainloader)
        iterator = tqdm(loader, desc='Batch', leave=False, position=2)
        return iterator

    def run(self):
        epoch_iter = self.get_epoch_iterator()
        for epoch in epoch_iter:
            self.train(epoch)
            self.test(epoch)
            self.save(epoch)

    def train(self, epoch):
        for m in self.models:
            m.train()

        if not self.distill_fixed_layer:
            tqdm.write('Randomly choosing a layer for distillation...')
            self.distill_cfg['layer'] = random.randint(1, self.depth)

        losses = [0 for i in range(len(self.models))]
        
        batch_iter = self.get_batch_iterator()
        for batch_idx, (si2, sl2, ti2, tl2) in enumerate(batch_iter):
            BS = len(si2) // 2
            si2, sl2, ti2, tl2 = si2.cuda(), sl2.cuda(), ti2.cuda(), tl2.cuda()
            si, sl = si2[:BS], sl2[:BS]
            ti, tl = ti2[:BS], tl2[:BS]
            if self.plus_adv:
                adv_inputs_list = []
                if self.combine_type == 0:
                    ensemble = Ensemble_max(self.models)
                else:
                    ensemble = Ensemble(self.models)
                adv_inputs = Linf_PGD(ensemble, si2[BS:], sl2[BS:], **self.attack_cfg)
                
            distilled_data_list = []
            for m in self.models:
                temp = Linf_distillation(m, si, ti, **self.distill_cfg)
                distilled_data_list.append(temp)

                if self.plus_adv:
                    temp = Linf_PGD(m, si, sl, **self.attack_cfg)
                    adv_inputs_list.append(temp)
                    
            r_loss = 0
            adv_loss = 0
            loss_dvr = 0
            ensemble_probs = 0
            ensemble_probs_part = 0
            outs_arr_sm = []
            outs_arr_tar = []
            outs_arr = []
            outs_arr_sm_copy = []
            
            for i, m in enumerate(self.models):
                loss = 0

                for j, distilled_data in enumerate(distilled_data_list):
                    if i == j:
                        continue

                    outputs = m(distilled_data)
                    loss += self.criterion(outputs, sl)
                
                if self.plus_adv:
                    outputs = m(adv_inputs_list[i])
                    loss = self.coeff * loss + self.criterion(outputs, sl)

                losses[i] += loss.item()
                loss_dvr += loss.item()
                self.optimizers[i].zero_grad()
                loss.backward()
                self.optimizers[i].step()     
                
                
                x_idx = torch.from_numpy(np.arange(adv_inputs.size(0))).cuda()
                outputs = m(adv_inputs)
                y_pred = F.softmax(outputs, dim=-1)
                outs_arr_sm.append(y_pred)
                outs_arr_sm_copy.append(y_pred.detach().clone())
                outs_arr_sm_copy[-1][x_idx, sl2[BS:]] = 0
                outs_arr.append(outputs)
                outs_arr_tar.append(y_pred[x_idx, sl2[BS:]].detach().clone().view(-1, 1))
                if self.combine_type == 0:
                    if i == 0:
                        ensemble_probs = y_pred
                    else:
                        ensemble_probs = torch.max(ensemble_probs, y_pred)
                else:
                    ensemble_probs += y_pred / len(self.models)
            
            _, preds = ensemble_probs.max(1)
            mask = ~preds.eq(sl2[BS:])
            outs_arr_tar = torch.cat(outs_arr_tar, dim=-1)
            outs_arr_sm_copy = torch.cat(outs_arr_sm_copy, dim=-1)
            outs_arr_sm = torch.cat(outs_arr_sm, dim=-1)
            idx_sort = torch.argsort(outs_arr_tar, dim=-1)
            
            probs = torch.zeros_like(outs_arr_tar).float()
            for j in range(len(self.models)):
                probs[x_idx, idx_sort[:, j]] = 2 ** j
            sel_branch = torch.multinomial(probs, 1, replacement=False)
            for i in range(len(self.models)):
                mask_partial =  (sel_branch == i).sum(dim=1).bool() & mask
                if mask_partial.sum().item() > 0:
                    adv_loss += self.criterion_sum(outs_arr[i][mask_partial], sl2[BS:][mask_partial]) / len(preds)
            if self.reg_type == 0:
                for j in range(len(self.models)):
                    _, idx_max_rest_sm = outs_arr_sm_copy[:, j*self.num_classes:(j+1)*self.num_classes].max(1)
                    idx_max_rest_sm = idx_max_rest_sm.detach()
                    if mask.sum().item() > 0:
                        r_loss += (-torch.log(1 - torch.clip(outs_arr_sm[x_idx, idx_max_rest_sm+j*self.num_classes][mask], 0, 0.990))).sum() / len(mask)
            else:
                _, idx_max_rest_sm = outs_arr_sm_copy.max(1)
                r_loss += (-torch.log(1 - torch.clip(outs_arr_sm[x_idx, idx_max_rest_sm][mask], 0., 0.990)) * 2).sum() /len(mask)              
                
            for i, m in enumerate(self.models):
                self.optimizers[i].zero_grad()
            loss = self.alpha * adv_loss + self.beta * r_loss
            loss.backward()
            for i, m in enumerate(self.models):
                self.optimizers[i].step()   
                
            if batch_idx % 10 == 0:
                print(self.alpha * adv_loss.item(), self.beta * r_loss.item(), loss_dvr)
            if batch_idx == 1:
                break
                
        #for i in range(len(self.models)):
        #    self.schedulers[i].step()

        print_message = 'Epoch [%3d] | ' % epoch
        for i in range(len(self.models)):
            print_message += 'Model{i:d}: {loss:.4f}  '.format(
                i=i+1, loss=losses[i]/(batch_idx+1))
        tqdm.write(print_message)

        loss_dict = {}
        for i in range(len(self.models)):
            loss_dict[str(i)] = losses[i]/len(self.trainloader)
        print('train/loss', loss_dict, epoch)

    def test(self, epoch):
        for m in self.models:
            m.eval()

        ensemble = Ensemble(self.models)

        loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for _, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.cuda(), targets.cuda()

                outputs = ensemble(inputs)
                loss += self.criterion(outputs, targets).item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()

                total += inputs.size(0)

        print('test/ensemble_loss', loss/len(self.testloader), epoch)
        print('test/ensemble_acc', 100*correct/total, epoch)

        print_message = 'Evaluation  | Ensemble Loss {loss:.4f} Acc {acc:.2%}'.format(
            loss=loss/len(self.testloader), acc=correct/total)
        tqdm.write(print_message)

    def save(self, epoch):
        state_dict = {}
        for i, m in enumerate(self.models):
            state_dict['model_%d'%i] = m.state_dict()
        if epoch % 5 == 0:
            torch.save(state_dict, os.path.join(self.save_root, 'epoch_cifar10_igat_dverge_%.4f_%d.pth'%(self.lr, epoch)))


def get_args():
    parser = argparse.ArgumentParser(description='CIFAR10 DVERGE Training of Ensemble', add_help=True)
    arguments.model_args(parser)
    arguments.data_args(parser)
    arguments.base_train_args(parser)
    arguments.dverge_train_args(parser)
    arguments.igat_train_args(parser)
    args = parser.parse_args()
    return args


def main():
    # get args
    args = get_args()
    # set up gpus
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    assert torch.cuda.is_available()

    # set up writer, logger, and save directory for models
    save_root = os.path.join('./checkpoints', 
        'dverge', 'seed_{:d}'.format(args.seed), '{:d}_{:s}{:d}_eps_{:.2f}'.format(
            args.model_num, args.arch, args.depth, args.distill_eps)
    )
    if args.distill_fixed_layer:
        save_root += '_fixed_layer_{:d}'.format(args.distill_layer)
    if args.plus_adv:
        print("plus_adv:", args.plus_adv)
        save_root += '_plus_adv_coeff_{:.1f}'.format(args.dverge_coeff)

    if not os.path.exists(save_root):
        os.makedirs(save_root)
    else:
        print('*********************************')
        print('* The checkpoint already exists *')
        print('*********************************')

    # set up random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    models = utils.get_models_cifar10(args, train=True, as_ensemble=False, model_file=args.model_file, leaky_relu=False)

    # get data loaders
    trainloader, testloader = utils.get_loaders_cifar10(args)

    # get optimizers and schedulers
    optimizers = utils.get_optimizers(args, models)
    #schedulers = utils.get_schedulers(args, optimizers)

    # train the ensemble
    trainer = DVERGE_Trainer(models, optimizers, None, trainloader, testloader, save_root, **vars(args))
    trainer.run()


if __name__ == '__main__':
    main()

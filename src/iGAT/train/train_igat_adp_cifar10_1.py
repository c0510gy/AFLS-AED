import os, sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
sys.path.append(os.getcwd())
import json, argparse, random
from tqdm import tqdm

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter

import arguments, utils
from models.ensemble import Ensemble, Ensemble_max
from distillation import Linf_PGD
import numpy as np

class ADP_Trainer():
    def __init__(self, models, trainloader, testloader,
                 save_root=None, **kwargs):
        self.models = models
        self.epochs = kwargs['epochs']
        self.trainloader = trainloader
        self.testloader = testloader
        self.reg_type = kwargs['reg_type']
        self.combine_type = kwargs['combine_type']
        self.igat_alpha = kwargs['igat_alpha']
        self.igat_beta = kwargs['igat_beta']
        self.alloc_type = 0
        self.save_root = save_root

        self.log_offset = 1e-20
        self.det_offset = 1e-6
        self.alpha = kwargs['alpha']
        self.beta = kwargs['beta']
        self.num_classes = kwargs['num_classes']
        self.best_acc = 0.0
        self.acc = 0.
        self.lr = kwargs['lr']

        params = []
        for m in self.models:
            params += list(m.parameters())
        self.optimizer = optim.SGD(params, lr=self.lr, weight_decay=1e-4, momentum=0.9)
        #self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=kwargs['sch_intervals'], gamma=kwargs['lr_gamma'])
        
        self.criterion = nn.CrossEntropyLoss()
        self.criterion_sum = nn.CrossEntropyLoss(reduction="sum")
        self.plus_adv = kwargs['plus_adv']
        if self.plus_adv:
            self.attack_cfg = {'eps': kwargs['adv_eps'], 
                               'alpha': kwargs['adv_alpha'],
                               'steps': kwargs['adv_steps'],
                               'is_targeted': False,
                               'rand_start': True
                              }
    
    def get_epoch_iterator(self):
        iterator = tqdm(list(range(1,self.epochs+1)), total=self.epochs, desc='Epoch',
                        leave=False, position=1)
        return iterator

    def get_batch_iterator(self):
        iterator = tqdm(self.trainloader, desc='Batch', leave=False, position=2)
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

        losses = 0
        ce_losses = 0
        ee_losses = 0
        det_losses = 0
        adv_losses = 0
        r_losses = 0
        batch_iter = self.get_batch_iterator()
        for batch_idx, (inputs_all, targets_all) in enumerate(batch_iter):
            inputs_all, targets_all = inputs_all.cuda(), targets_all.cuda()
            N = len(inputs_all) // 2
            inputs, targets = inputs_all[:N], targets_all[:N]
            inputs2, targets2 = inputs_all[N:], targets_all[N:]

            if self.plus_adv:
                if self.combine_type == 0:
                    ensemble = Ensemble(self.models)
                else:
                    ensemble = Ensemble_max(self.models)
                adv_inputs = Linf_PGD(ensemble, inputs2, targets2, **self.attack_cfg)

            image_per = torch.cat([inputs, adv_inputs], dim=0)
            x_idx = torch.from_numpy(np.arange(image_per.size(0))).cuda()
            # one-hot label
            y_true = torch.zeros(inputs_all.size(0), self.num_classes).cuda()
            y_true.scatter_(1, targets_all.view(-1,1), 1)
    
            ce_loss = 0
            adv_loss = 0
            r_loss = 0
            mask_non_y_pred = []
            ensemble_probs = 0
            outs_arr_sm = []
            outs_arr_tar = []
            outs_arr = []
            outs_arr_sm_copy = []
            for i, m in enumerate(self.models):
                outputs = m(image_per)
                y_pred = F.softmax(outputs, dim=-1)
                outs_arr_sm.append(y_pred)
                outs_arr_sm_copy.append(y_pred.detach().clone())
                outs_arr_sm_copy[-1][x_idx, targets_all] = 0
                outs_arr.append(outputs)
                outs_arr_tar.append(y_pred[x_idx, targets_all].detach().clone().view(-1, 1))

                # for log_det
                bool_R_y_true = torch.eq(torch.ones_like(y_true) - y_true, torch.ones_like(y_true)) # batch_size X (num_class X num_models), 2-D
                mask_non_y_pred.append(torch.masked_select(y_pred, bool_R_y_true).reshape(-1, self.num_classes-1)) # batch_size X (num_class-1) X num_models, 1-D

                # for ensemble entropy
                if self.combine_type == 0:
                    ensemble_probs += y_pred / len(self.models)
                else:
                    if i == 0:
                        ensemble_probs = y_pred
                    else:
                        ensemble_probs = torch.max(ensemble_probs, y_pred)
            
            _, preds = ensemble_probs.max(1)
            
            mask = ~preds.eq(targets_all)
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
                    ce_loss += self.criterion_sum(outs_arr[i][mask_partial], targets_all[mask_partial]) / len(preds)

            if self.reg_type == 0:
                for j in range(len(self.models)): 
                    _, idx_max_rest_sm = outs_arr_sm_copy[:, j*self.num_classes:(j+1)*self.num_classes].max(1)
                    idx_max_rest_sm = idx_max_rest_sm.detach()
                    if mask.sum().item() > 0:
                        r_loss += (-torch.log(1 - torch.clip(outs_arr_sm[x_idx, idx_max_rest_sm+j*self.num_classes][mask], 0, 0.990))).sum() / len(mask)
            else:
                _, idx_max_rest_sm = outs_arr_sm_copy.max(1)
                r_loss += (-torch.log(1 - torch.clip(outs_arr_sm[x_idx, idx_max_rest_sm][mask], 0., 0.990)) * 2).sum() /len(mask) 

            ensemble_entropy = torch.sum(-torch.mul(ensemble_probs, torch.log(ensemble_probs + self.log_offset)), dim=-1).mean()

            mask_non_y_pred = torch.stack(mask_non_y_pred, dim=1)
            assert mask_non_y_pred.shape == (inputs_all.size(0), len(self.models), self.num_classes-1)
            mask_non_y_pred = mask_non_y_pred / torch.norm(mask_non_y_pred, p=2, dim=-1, keepdim=True) # batch_size X num_model X (num_class-1), 3-D
            matrix = torch.matmul(mask_non_y_pred, mask_non_y_pred.permute(0, 2, 1)) # batch_size X num_model X num_model, 3-D
            log_det = torch.logdet(matrix+self.det_offset*torch.eye(len(self.models), device=matrix.device).unsqueeze(0)).mean() # batch_size X 1, 1-D
            
            loss = self.igat_alpha * ce_loss + self.igat_beta * r_loss - self.alpha * ensemble_entropy - self.beta * log_det 
            losses += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()         
        
            if batch_idx % 10 == 0:
                print(self.igat_alpha * ce_loss.item(), self.igat_beta * r_loss.item(), self.alpha * ensemble_entropy.item(), self.beta * log_det.item())
            
            ce_losses += self.igat_alpha * ce_loss.item()
            ee_losses +=  -self.alpha * ensemble_entropy.item()
            det_losses += -self.beta * log_det.item()
            r_losses += self.igat_beta * r_loss.item()
	    
            if batch_idx == 50:
                break
        #self.scheduler.step()

        print_message = 'Epoch [%3d] | ' % epoch
        for i in range(len(self.models)):
            print_message += 'Model{i:d}: {loss:.4f}  '.format(
                i=i+1, loss=losses/(batch_idx+1))
        tqdm.write(print_message)

        print('train/ce_loss', ce_losses/len(self.trainloader), epoch)
        print('train/ee_loss', ee_losses/len(self.trainloader), epoch)
        print('train/det_loss', det_losses/len(self.trainloader), epoch)
        print('train/r_loss', r_losses/len(self.trainloader), epoch)

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
            torch.save(state_dict, os.path.join(self.save_root, 'epoch_cifar10_igat_adp_%.4f_%d.pth'%(self.lr, epoch)))


def get_args():
    parser = argparse.ArgumentParser(description='CIFAR10 ADP Training of Ensemble', add_help=True)
    arguments.model_args(parser)
    arguments.data_args(parser)
    arguments.base_train_args(parser)
    arguments.adp_train_args(parser)
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
        'adp', 'seed_{:d}'.format(args.seed), '{:d}_{:s}{:d}'.format(
          args.model_num, args.arch, args.depth)
    )

    if args.plus_adv:
        print("plus_adv:", args.plus_adv)
        save_root += '_plus_adv'
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    else:
        print('*********************************')
        print('* The checkpoint already exists *')
        print('*********************************')

    # set up random seed
    torch.manual_seed(args.seed)

    models = utils.get_models_cifar10(args, train=True, as_ensemble=False, model_file=args.model_file, leaky_relu=False)

    # get data loaders
    trainloader, testloader = utils.get_loaders_cifar10(args)

    # train the ensemble
    trainer = ADP_Trainer(models, trainloader, testloader, save_root, **vars(args))
    trainer.run()


if __name__ == '__main__':
    main()

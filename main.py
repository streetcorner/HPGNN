import yaml
import sys
import os

import optuna

project_root = os.path.dirname(os.path.abspath(__file__)) 
project_root = os.path.abspath(project_root)  

sys.path.append(project_root)


from utils.dataset_utils import DataLoader, random_planetoid_splits
from utils.param_utils import *
import torch
import torch.nn.functional as F 
from tqdm import tqdm 
import numpy as np
from torch_geometric.utils import homophily


def RunExp(args, dataset, data, Net, percls_trn, val_lb):
    def train(model, optimizer, data, dprate):
        model.train()
        optimizer.zero_grad()
        out = model(data)[data.train_mask]
        nll = F.nll_loss(out, data.y[data.train_mask])
        loss = nll
        loss.backward()
        optimizer.step()
        del out
    def test(model, data, order):
        model.eval()
        logits, accs, losses, preds = model(data), [], [], []
    
        """
        data = {
            'train_mask': torch.tensor([True, False, True]),  
            'val_mask': torch.tensor([False, True, False]),  
            'test_mask': torch.tensor([False, False, True])   
        }
        """
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            loss = F.nll_loss(model(data)[mask], data.y[mask])
            preds.append(pred.detach().cpu()) 
            accs.append(acc) 
            losses.append(loss.detach().cpu()) 
        return accs, preds, losses 

    appnp_net = Net(dataset, args)
    torch.cuda.set_device(args.cuda)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    permute_masks = random_planetoid_splits
    data = permute_masks(data, dataset.num_classes, percls_trn, val_lb, args.seed)
    model, data = appnp_net.to(device), data.to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history, val_acc_history = [], []  
    beta = {}  
    for i in range(args.Order): 
        beta[i] = []
    order=args.Order
    for epoch in range(args.epochs):
        train(model, optimizer, data, args.dprate)
        
        [train_acc, val_acc, tmp_test_acc], preds, [
            train_loss, val_loss, tmp_test_loss] = test(model, data,order)

        if val_loss < best_val_loss:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            if args.net == 'HPGNN':
            
                for tmp_order in range(0, order):
                    TEST = appnp_net.hgc[tmp_order].fW.clone()  
                    Alpha = TEST.detach().cpu().numpy()  
                    beta[tmp_order] = abs(Alpha)  

        if epoch >= 0:   
            val_loss_history.append(val_loss)  
            val_acc_history.append(val_acc)   
            
            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.tensor(
                    val_loss_history[-(args.early_stopping + 1):-1]) 
                if val_loss > tmp.mean().item():   
                    break
        return test_acc, best_val_acc, beta



if __name__ == '__main__':

    args = parse_args() 
    #10 fixed seeds for splits
    SEEDS=[1941488137,4198936517,983997847,4023022221,4019585660,2108550661,1648766618,629014539,3212139042,2424918363]
    
    Net = get_net(args.net)  
    print(f"use net {args.net}")
    dataset, data = DataLoader(args.dataset, args) 
    RPMAX = args.RPMAX  
    homo = homophily(data.edge_index, data.y)  
    beta = {}
    alpha = args.alpha  
    train_rate = args.train_rate  
    val_rate = args.val_rate  
    percls_trn = int(round(train_rate * len(data.y) / dataset.num_classes))  
    val_lb = int(round(val_rate * len(data.y)))  
    TrueLBrate = (percls_trn * dataset.num_classes + val_lb) / len(data.y)  
    print('True Label rate: ', TrueLBrate)
    args.C = len(data.y.unique())  
    args.beta = beta

    Results0 = []  
    Result_test = []
    Result_val = []
   
    for RP in tqdm(range(RPMAX)):
        args.seed=SEEDS[RP % 10]
        test_acc, best_val_acc, beta = RunExp(args, dataset, data, Net, percls_trn, val_lb)
        Results0.append([test_acc, best_val_acc])  
        Result_test.append(test_acc)
        Result_val.append(best_val_acc)
        
    test_acc_mean, val_acc_mean = np.mean(Results0, axis=0) * 100 
    test_acc_std = np.sqrt(np.var(Results0, axis=0)[0]) * 100 
    
    print(f'{args.net} on dataset {args.dataset}, in {RPMAX} repeated experiment:')
    print(f'test acc mean = {test_acc_mean:.4f} \t test acc std = {test_acc_std:.4f} \t val acc mean = {val_acc_mean:.4f}')
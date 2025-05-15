#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8


import optuna
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
#from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn import Parameter
from torch.nn import Linear
from torch_geometric.nn import GATConv, GCNConv, ChebConv
from torch_geometric.nn import JumpingKnowledge
from torch_geometric.nn import MessagePassing, APPNP
from torch_sparse import matmul, SparseTensor


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class HPGNN_prop(MessagePassing):
    '''
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Order=2, bias=True, **kwargs):
        super(HPGNN_prop, self).__init__(aggr='add', **kwargs)
        self.K = K  
        self.alpha = alpha  
        self.Order = Order  
        self.fW = Parameter(torch.Tensor(self.K + 1)) 
        self.reset_parameters()
        #Parameter containing:tensor([5.7000e-01, 2.4510e-01, 1.0539e-01, 4.5319e-02, 1.9487e-02, 8.3795e-03,
        # 3.6032e-03, 1.5494e-03, 6.6623e-04, 2.8648e-04, 2.1611e-04],requires_grad=True)
    def reset_parameters(self):
        torch.nn.init.zeros_(self.fW)  
        for k in range(self.K + 1):
            self.fW.data[k] = self.alpha * (1 - self.alpha) ** k  
        self.fW.data[-1] = (1 - self.alpha) ** self.K  

    def forward(self, x, PPM):
        hidden = x * (self.fW[0])  
        for k in range(self.K):
            x = matmul(PPM, x, reduce=self.aggr)  
            beta = self.fW[k + 1] 
            hidden = hidden + beta * x  
        return hidden  

    def __repr__(self):
        return '{}(Order={}, K={}, filterWeights={})'.format(self.__class__.__name__, self.Order, self.K, self.fW)


class HPGNN(torch.nn.Module):  
    def __init__(self, dataset, args):  
        super(HPGNN, self).__init__() 
        self.Order = args.Order 
        self.lin_in = nn.ModuleList()  
        self.hgc = nn.ModuleList() 

        for i in range(args.Order): 
            self.lin_in.append(Linear(dataset.num_features, args.hidden)) 
            self.hgc.append(HPGNN_prop(args.K, args.alpha, args.Order)) 
        self.lin_out = Linear(args.hidden * args.Order, dataset.num_classes) 
        self.dprate = args.dprate  
        self.dropout = args.dropout 
    
    def forward(self, data):
        x, PPM = data.x, data.PPM 
        x_concat = torch.tensor([]).to(device)
        for i in range(self.Order):
            xx = F.dropout(x, p=self.dropout, training=self.training)
            xx = self.lin_in[i](xx) 
            if self.dprate > 0.0:
                xx = F.dropout(xx, p=self.dprate, training=self.training)            
            xx = self.hgc[i](xx, PPM[i + 1])
            x_concat = torch.cat((x_concat, xx), 1) 

        x_concat = F.dropout(x_concat, p=self.dropout, training=self.training)
        x_concat = self.lin_out(x_concat)
        return F.log_softmax(x_concat, dim=1)

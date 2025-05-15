#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#包含了一些参数相关的工具函数或类。

import argparse
from models.HPGNN_model import HPGNN
from models.benchmarks import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='texas', help='dataset name')

    parser.add_argument('--RPMAX', type=int, default=100, help='repeat times')
    parser.add_argument('--Order', type=int, default=2, help='max simplix dimension')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--early_stopping', type=int, default=200)

    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--dprate', type=float, default=0.3) #隐藏层的丢弃
    parser.add_argument('--dropout', type=float, default=0.5) #对输入和输出的丢弃
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--train_rate', type=float, default=0.6)
    parser.add_argument('--val_rate', type=float, default=0.2)
    parser.add_argument('--hidden', type=int, default=32)
    
    parser.add_argument('--print_freq', type=int, default=20)
    parser.add_argument('--cuda', type=int, default=3)
    parser.add_argument('--alphappr', type=float, default=0.15)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--net', type=str,
                        choices=['GCN', 'GAT', 'APPNP', 'ChebNet', 'JKNet', 'GPRGNN', 'BernNet', 'HPGNN'],
                        default='HPGNN',
                        )
    """
    The following arguments are used in the benchmarks!
    """
    parser.add_argument('--heads', default=8, type=int)
    parser.add_argument('--output_heads', default=1, type=int)
    parser.add_argument('--C', type=int)
    parser.add_argument('--Init', type=str, default='PPR')
    parser.add_argument('--beta', default=None)
    parser.add_argument('--ppnp', choices=['PPNP', 'GPR_prop'], default='GPR_prop')
    parser.add_argument('--Bern_lr', type=float, default=0.002, help='learning rate for BernNet propagation layer.')
    args = parser.parse_args()
    return args

def get_net(gnn_name):
    if gnn_name == 'GCN':
        Net = GCN_Net
    elif gnn_name == 'GAT':
        Net = GAT_Net
    elif gnn_name == 'APPNP':
        Net = APPNP_Net
    elif gnn_name == 'ChebNet':
        Net = ChebNet
    elif gnn_name == 'JKNet':
        Net = GCN_JKNet
    elif gnn_name == 'GPRGNN':
        Net = GPRGNN
    elif gnn_name == 'BernNet':
        Net = BernNet
    elif gnn_name == 'HPGNN':
        Net = HPGNN

    return Net
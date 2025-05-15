"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#包含用于加载数据集和进行数据集划分的工具函数。
"""

import torch

import os
import os.path as osp
import pickle
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Amazon
from torch_geometric.datasets import WikipediaNetwork
from torch_geometric.datasets import Actor
import pandas as pd
from torch_sparse import coalesce
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.utils.undirected import to_undirected
from torch_geometric.utils import homophily
import torch
import numpy as np
import networkx as nx
import torch_sparse as sp
from torch_sparse import SparseTensor, fill_diag, matmul, mul
from scipy.sparse import csr_matrix
from tqdm import tqdm #进度条
from utils.gen_ppr import power_push_sor
import sys



class dataset_heterophily(InMemoryDataset):
    def __init__(self, root='data/', name=None,
                 p2raw=None,
                 train_percent=0.01,
                 transform=None, pre_transform=None):

        existing_dataset = ['chameleon', 'film', 'squirrel']
        if name not in existing_dataset:
            raise ValueError(
                f'name of hypergraph dataset must be one of: {existing_dataset}')
        else:
            self.name = name

        self._train_percent = train_percent

        if (p2raw is not None) and osp.isdir(p2raw):
            self.p2raw = p2raw
        elif p2raw is None:
            self.p2raw = None
        elif not osp.isdir(p2raw):
            raise ValueError(
                f'path to raw hypergraph dataset "{p2raw}" does not exist!')

        if not osp.isdir(root):
            os.makedirs(root)

        self.root = root

        super(dataset_heterophily, self).__init__(
            root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])
        self.train_percent = self.data.train_percent

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        file_names = [self.name]
        return file_names

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        p2f = osp.join(self.raw_dir, self.name)
        with open(p2f, 'rb') as f:
            data = pickle.load(f)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


class WebKB(InMemoryDataset):
    r"""The WebKB datasets used in the
    `"Geom-GCN: Geometric Graph Convolutional Networks"
    <https://openreview.net/forum?id=S1e2agrFvS>`_ paper.
    Nodes represent web pages and edges represent hyperlinks between them.
    Node features are the bag-of-words representation of web pages.
    The task is to classify the nodes into one of the five categories, student,
    project, course, staff, and faculty.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Cornell"`,
            :obj:`"Texas"` :obj:`"Washington"`, :obj:`"Wisconsin"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = ('https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/'
           'master/new_data')

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in ['cornell', 'texas', 'washington', 'wisconsin']
        super(WebKB, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['out1_node_feature_label.txt', 'out1_graph_edges.txt']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # super().download()
        for name in self.raw_file_names:
            download_url(f'{self.url}/{self.name}/{name}', self.raw_dir)

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            data = f.read().split('\n')[1:-1]
            x = [[float(v) for v in r.split('\t')[1].split(',')] for r in data]
            x = torch.tensor(x, dtype=torch.float)

            y = [int(r.split('\t')[2]) for r in data]
            y = torch.tensor(y, dtype=torch.long)

        with open(self.raw_paths[1], 'r') as f:
            data = f.read().split('\n')[1:-1]
            data = [[int(v) for v in r.split('\t')] for r in data]
            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            # 有向图转无向图
            edge_index = to_undirected(edge_index)
            # 去除了图中的重复边和自环
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        data = Data(x=x, edge_index=edge_index, y=y)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


def graphLoader(name):
    project_root = osp.abspath(osp.join(osp.dirname(__file__), "../.."))
    root_path = osp.join(project_root, "HPGNN/data")


    if name in ['cora', 'citeseer', 'pubmed']:
        path = osp.join(root_path, name)
        print("download path=",path)
        dataset = Planetoid(path, name=name) 
        data = dataset[0] 
    elif name in ['computers', 'photo']:
        path = osp.join(root_path, name)
        dataset = Amazon(path, name, T.NormalizeFeatures())
        data = dataset[0]
    elif name in ['chameleon', 'squirrel']:
        # use everything from "geom_gcn_preprocess=False" and
        # only the node label y from "geom_gcn_preprocess=True"
        preProcDs = WikipediaNetwork(
            root=root_path, name=name, geom_gcn_preprocess=False, transform=T.NormalizeFeatures())
        dataset = WikipediaNetwork(
            root=root_path, name=name, geom_gcn_preprocess=True, transform=T.NormalizeFeatures())
        data = dataset[0]
        data.edge_index = preProcDs[0].edge_index
        # return dataset, data
    elif name in ['film']:
        dataset = Actor(root=osp.join(root_path, 'film'), transform=T.NormalizeFeatures())
        data = dataset[0]
    elif name in ['texas', 'cornell', 'wisconsin']:
        print("root_path=",root_path)
        dataset = WebKB(root=root_path, name=name, transform=T.NormalizeFeatures())
        data = dataset[0]
    else:
        raise ValueError(f'dataset {name} not supported in graphLoader')

    return data, dataset


#计算PPR矩阵
def cal_ppm(data,ppm_path,args):
    
    print("Preparing parameters for calucating ppr matix...")
    order=args.Order
    row, col,temp, adj_matrix_csr, data_values,clique = {},{},{},{},{},{}
    cliqueNum =[0] * (args.Order + 1) 
    for i in range(1, args.Order + 1): 
        row[i], col[i] ,clique[i] = [], [], []
        temp[i] = np.ones(i + 1, dtype=np.int) 
    row[1] = data.edge_index[0].numpy() 
    col[1] = data.edge_index[1].numpy()  
    edges = list(zip(row[1], col[1]))
    cliqueNum[0]=data.num_nodes
    cliqueNum[1]=len(edges)
    
    if(order > 1): 
        G = nx.Graph()
        G.add_nodes_from(range(data.num_nodes))
        G.add_edges_from(data.edge_index.numpy().transpose())
        itClique = nx.enumerate_all_cliques(G)
        nextClique = next(itClique) 
        while len(nextClique) <= 2: 
            nextClique = next(itClique)
      
        while len(nextClique) <= (order + 1):
            tmp_order = len(nextClique) - 1 
            clique[tmp_order].append(nextClique) 
            row[tmp_order].extend(nextClique)   
            col[tmp_order].extend(cliqueNum[tmp_order] * temp[tmp_order]) 
            cliqueNum[tmp_order] += 1  
            try:
                nextClique = next(itClique)
            except StopIteration:
                break
    '''
    edge_index = tensor([[0, 1, 2],
                [1, 2, 0]])
    '''
    for i in range(1, args.Order + 1):
        data_values[i] = np.ones(len(row[i]))
        if(i==1):
            adj_matrix_csr[i] = csr_matrix((data_values[i], (row[i], col[i])), shape=(cliqueNum[0], cliqueNum[0]))
        else: 
            adj_matrix_csr[i] = csr_matrix((data_values[i], (row[i], col[i])), shape=(cliqueNum[0], cliqueNum[i]))
    print("--------------------------------------------------------------")
    print("Calucating ppr matix...")

    alpha=args.alphappr
    eps=args.eps
    node_id=0
    '''
     print("edge_index",data.edge_index)
     edge_index tensor([[   0,    0,    0,  ..., 2707, 2707, 2707],
    '''
    all_nonzero_values,all_col_indices,all_row_indices = {},{},{}

    for i in range( 1,  order+1 ): 
        all_nonzero_values[i],all_col_indices[i],all_row_indices[i]=[],[],[]


    for node in tqdm(range(cliqueNum[0])):
        ppr=power_push_sor(adj_matrix_csr,clique,cliqueNum, node,eps,alpha, maxCliqueSize=order)
        
        '''
        #[7.36932797e-05 4.10629632e-05 8.97597657e-05 ... 3.00001000e+05 3.25436000e+05 4.00000000e+00]
        '''
    
        for i in range( 1,  order+1 ): 
            if(sum(ppr[i])==0): 
                break
            ppr[i]=torch.tensor(ppr[i], dtype=torch.float64)
            col_indices = torch.nonzero(ppr[i] > 0).squeeze() 
            nonzero_values = ppr[i][col_indices]  
            row_indices = torch.full_like(col_indices, node)   
            if nonzero_values.dim()==0:
                nonzero_values=nonzero_values.unsqueeze(0)
            if col_indices.dim()==0:
                col_indices=col_indices.unsqueeze(0)
            if row_indices.dim()==0:
                row_indices=row_indices.unsqueeze(0)

            all_nonzero_values[i].append(nonzero_values) #all_nonzero_indices=[tensor([   0,    1,  ..., 2707]), tensor([   0,    1,   ...,  2707])]
            all_col_indices[i].append(col_indices)  
            all_row_indices[i].append(row_indices)  
    PPM={} 
    for i in range( 1,  order + 1 ):
        if not all_nonzero_values[i]: 
            print(f"Warning: No nonzero values for order {i}")
            continue  
        cur_all_nonzero_values = torch.cat(all_nonzero_values[i])  
        cur_all_col_indices = torch.cat(all_col_indices[i])  
        cur_all_row_indices = torch.cat(all_row_indices[i])  
        
        cur_coo_indices = torch.stack([cur_all_row_indices, cur_all_col_indices], dim=0)    
        size = torch.Size([cliqueNum[0],cliqueNum[0]])  
        ppm_coo = torch.sparse_coo_tensor(cur_coo_indices, cur_all_nonzero_values, size)
        PPM[i] = SparseTensor.from_torch_sparse_coo_tensor(ppm_coo)
    data.PPM=PPM 
    
    torch.save(data.PPM, ppm_path) 
    '''
    data.PPM={1: SparseTensor(row=tensor([2707, 2707, 2707,  ..., 2707, 2707, 2707]),
             col=tensor([   0,    1,    2,  ..., 2702, 2706, 2707]),
             val=tensor([3.4068e-04, 1.3032e-04, 4.5543e-04,  ..., 8.9058e-06, 8.9555e-02,
                           2.0740e-01], dtype=torch.float64),
             size=(2708, 2708), nnz=2485, density=0.03%)}

    '''

def DataLoader(name, args):
    """
    加载图数据集，并根据需要计算高阶结构（如高阶拉普拉斯矩阵）。
    参数：
        - name: 数据集的名称（如 'cora', 'texas' 等）。
        - args: 包含超参数和配置的对象，通常包含网络模型名（args.net）、高阶结构阶数（args.Order）、扰动参数（args.rho）等。
    返回：
        - dataset: 加载的图数据集对象。
        - data: 数据集中的图数据，包含节点特征、边、标签等信息。
    """
   
    ppm_path = osp.join('data', name + '/PPM_' + name + '.pt') 

    
    if name in ['cora', 'citeseer', 'pubmed', 'computers', 'photo', 'chameleon', 'squirrel', 'film', 'texas', 'cornell', 'wisconsin']:
        data, dataset = graphLoader(name)
        print(f"{name} inn")
    elif name in ['Texas_null']:
        print(f"Texas_null in")
        """
        Texas_null is a null model to test different effect of higher-order structures
        """
        name = 'Texas'
        path = '../data/nullModel_Texas/'
        dataset = WebKB(root='../data/',
                        name=name, transform=T.NormalizeFeatures())
        data = dataset[0]
        change = args.rho

        G = nx.Graph()
        graph_edge_list = []
        dataset_path = osp.join('..','data','nullModel_'+name, name+'_1_generate' + change + '.txt')
        lines = pd.read_csv(dataset_path)

        G.add_edge(int(lines.keys()[0].split(' ')[0]), int(lines.keys()[0].split(' ')[1]))
        graph_edge_list.append([int(lines.keys()[0].split(' ')[0]), int(lines.keys()[0].split(' ')[1])])
        graph_edge_list.append([int(lines.keys()[0].split(' ')[1]), int(lines.keys()[0].split(' ')[0])])
        for line in lines.values:
            G.add_edge(int(line[0].split(' ')[0]), int(line[0].split(' ')[1]))
            graph_edge_list.append([int(line[0].split(' ')[0]), int(line[0].split(' ')[1])])
            graph_edge_list.append([int(line[0].split(' ')[1]), int(line[0].split(' ')[0])])

        data.edge_index = torch.tensor(graph_edge_list).H
        # print("当前路径:", os.getcwd())
        print(" ppm_path:", ppm_path)
    else:
        raise ValueError(f'dataset {name} not supported in dataloader')

    print(data)
    print(dataset)
    if (args.net =='HPGNN'):
        '''
        data.PPM={1: SparseTensor(row=tensor([2707, 2707, 2707,  ..., 2707, 2707, 2707]),
             col=tensor([   0,    1,    2,  ..., 2702, 2706, 2707]),
             val=tensor([3.4068e-04, 1.3032e-04, 4.5543e-04,  ..., 8.9058e-06, 8.9555e-02,
                           2.0740e-01], dtype=torch.float64),
             size=(2708, 2708), nnz=2485, density=0.03%)}
        '''

        '''
        PPM 换L
        '''
        if osp.exists(ppm_path):
            print("当前路径:", os.getcwd())
            print(" ppm_path:", ppm_path)
            data.PPM = torch.load(ppm_path)  
            if len(data.PPM) < args.Order: 
                print("阶数不够，重新计算")
                cal_ppm(data,ppm_path,args)
        else: 
            cal_ppm(data,ppm_path,args)

        print(F"{data.PPM}")
    print(f"load data {name} finished")
      

    # 计算图的同配性
    homo = homophily(data.edge_index, data.y)
    print("Home:", homo)
    print("Finish load data!")
    return dataset, data



def index_to_mask(index, size):
    # mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask

def random_planetoid_splits(data, num_classes, percls_trn=20, val_lb=500, seed=12134, Flag=0):
    # 设置新的随机划分，参数说明：
    # percls_trn: 每个类别用于训练的样本数
    # val_lb: 验证集的样本数量
    # Flag: 控制验证集和测试集的划分方式

    index=[i for i in range(0,data.y.shape[0])]  
    train_idx=[]  
    rnd_state = np.random.RandomState(seed)  # 根据固定的随机种子seed,创建一个随机数生成器,保证每次划分的一致性
    for c in range(num_classes):  # 遍历每个类别
        class_idx = np.where(data.y.cpu() == c)[0]  
        if len(class_idx)<percls_trn:  
            train_idx.extend(class_idx)  
        else:  
            train_idx.extend(rnd_state.choice(class_idx, percls_trn,replace=False))  
    rest_index = [i for i in index if i not in train_idx]  
    val_idx=rnd_state.choice(rest_index,val_lb,replace=False)  
    test_idx=[i for i in rest_index if i not in val_idx]  

    data.train_mask = index_to_mask(train_idx,size=data.num_nodes)  
    data.val_mask = index_to_mask(val_idx,size=data.num_nodes)  
    data.test_mask = index_to_mask(test_idx,size=data.num_nodes)  

    return data  





class MyTransform(object):
    def __call__(self, data):
        data.y = data.y[:, 8]  # Specify target: 0 = mu
        return data


class MyFilter(object):
    def __call__(self, data):
        return not (data.num_nodes == 7 and data.num_edges == 12) and \
               data.num_nodes < 450

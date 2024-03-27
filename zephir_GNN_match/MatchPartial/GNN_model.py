import re
import math
import os.path as osp
import time
from itertools import chain
import h5py
import pandas as pd
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import cdist

from torch.nn import BCEWithLogitsLoss, Conv1d, MaxPool1d, ModuleList
import torch.nn as nn
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, GCNConv, GATConv, SortAggregation, PMLP
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import k_hop_subgraph, to_scipy_sparse_matrix
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix,roc_auc_score





class DGCNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, num_features, GNN, conv1d_channels, use_edge_attr, k):
        super().__init__()

        self.convs = ModuleList()
        
        self.convs.append(GNN(num_features, hidden_channels))
        for i in range(0, num_layers - 1):
            self.convs.append(GNN(hidden_channels, hidden_channels))
        self.convs.append(GNN(hidden_channels, 1))


        total_latent_dim = hidden_channels * num_layers + 1
       
        conv1d_kws = [total_latent_dim, 1] 

        self.conv1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0],
                            conv1d_kws[0])

        self.use_edge_attr = use_edge_attr
       
        self.k = k
        
    def forward(self, x, edge_index, edge_attr):
        xs = [x]
        
        
        for conv in self.convs:
            # xs += [conv(xs[-1], edge_index).tanh()]
            if self.use_edge_attr :
                xs += [conv(xs[-1], edge_index, edge_attr)]
            elif not self.use_edge_attr :
                xs += [conv(xs[-1], edge_index)]
           

        x = torch.cat(xs[self.k:], dim=-1)

        x = x.unsqueeze(1)  # [num_graphs, 1, k * hidden]
        x = self.conv1(x)
        
        return x[:,:,0]



def weights_init_uniform(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # apply a uniform distribution to the weights and a bias=0
        m.weight.data.uniform_(0.0, 1.0)
        m.bias.data.fill_(0)
        
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
        
        
        
        
        
class NodeLevelGNN(torch.nn.Module):
    def __init__(self, nearby_search_num, with_AM, device, train_val):
        '''
        nearby_search_num: the number of nearby nodes cross frame that are taken into consideration
        with_AM: if put the ground truth into training
        device: 'cuda' or 'cpu'
        train_val: if 'train', will train only on the positive datasets, else for evaluation do prediction for all
        '''
        super(NodeLevelGNN, self).__init__()
  
        self.encoder = DGCNN(hidden_channels=64, num_layers=5, num_features=10, GNN=GATConv, conv1d_channels = [16,32],  use_edge_attr = True, k=0)
        self.encoder2 = DGCNN(hidden_channels=8, num_layers=3, num_features=10, GNN=GATConv, conv1d_channels = [16,64],  use_edge_attr = False, k=0)
        
        
        self.mlp = MLP([1, 32, 1], dropout=1e-10, norm=None)
        self.m = nn.Conv1d(32, 10, 1, stride=1)
        self.m2 = nn.Conv1d(10, 2, 1, stride=1)
        # self.m3 = nn.Conv1d(4, 2, 1, stride=1)
        self.nearby_search_num = nearby_search_num
        self.with_AM = with_AM
        self.device = device
        self.train_val = train_val
        if self.train_val!= 'train':
            self.with_AM = False 
        
  
    
    def each_graph(self, x, edge_index, edge_attr):
        emb = self.encoder(x, edge_index, edge_attr)

        emb2 = self.encoder2(x, edge_index, edge_attr)
        return torch.cat((emb2,emb),dim = 1)
  
        
        # return emb
    
    
    def get_ground_truth(self,data1,data2):
        eff_label = (data1.y.unsqueeze(1) + data2.y.unsqueeze(0)) > 0  ## to exclude label=0
        AM = ((data1.y.unsqueeze(1) - data2.y.unsqueeze(0)) == 0)*(eff_label)*1
        return AM

    
    def get_AM_mask(self,data1,data2,with_AM):
        # self.nearby_search_num = 10
        coord = data1.x[:,0:3]
        coords2 = data2.x[:,0:3]
        distance_matrix = torch.cdist(coord,coords2)
        # nearest_indices = np.argsort(distance_matrix, axis=1)[:, :nearby_search_num]
        nearest_indices = torch.argsort(distance_matrix, dim=1)[:, :self.nearby_search_num]
        mask = torch.zeros_like(distance_matrix, dtype=bool).to(self.device)
        
        for row_idx, col_indices in enumerate(nearest_indices):
            mask[row_idx, col_indices] = 1

        if with_AM:
            AM = self.get_ground_truth(data1,data2)
            AM_mask = ((mask) + 1*(AM>0))>0
        else:
            AM_mask = mask>0
        if self.train_val== 'train':
            AM_mask[data1.y==0] = 0
        # elif train_val == 'val':
            
        return AM_mask
    
    
    def single_loss(self,data1,data2):
        
        # ind = data1.y>0
        pred1 = self.each_graph(data1.x[:,0:10], data1.edge_index, data1.edge_attr[:,0:4])

        pred2 = self.each_graph(data2.x[:,0:10], data2.edge_index, data2.edge_attr[:,0:4])
       
        AM_mask = self.get_AM_mask(data1,data2,self.with_AM)
 
        potential_edge = torch.transpose(torch.stack(torch.where(AM_mask>0)),1,0)

        
        embedding1 = pred1[potential_edge[:,0]]
        embedding2 = pred2[potential_edge[:,1]]
    
    
        diff = torch.abs(embedding1 - embedding2)

        


        sim = self.mlp(diff[:,:,None])
        
        sim = self.m(sim)
        sim = self.m2(sim)
  
        
        return sim[:,:,0]
    

    def forward(self,data1,data2):
        return self.single_loss(data1,data2)
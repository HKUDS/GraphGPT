import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
import json
import copy
from transformers import AutoTokenizer 
import transformers
from transformers.configuration_utils import PretrainedConfig
import os

def gcn_conv(h, edge_index):
    # print(edge_index)
    N, node_feas = h.shape
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=N)
    
    src, dst = edge_index
    deg = degree(dst, num_nodes=N)

    deg_src = deg[src].pow(-0.5) 
    deg_src.masked_fill_(deg_src == float('inf'), 0)
    deg_dst = deg[dst].pow(-0.5)
    deg_dst.masked_fill_(deg_dst == float('inf'), 0)
    edge_weight = deg_src * deg_dst

    a = torch.sparse_coo_tensor(edge_index, edge_weight, torch.Size([N, N])).t()
    rows, cols = edge_index
    edge_msg = h[rows, :] * torch.unsqueeze(edge_weight, dim=-1)
    col_embeds = h[cols, :]
    tem = torch.zeros([N, node_feas]).to(edge_msg.device)
    rows = rows.to(edge_msg.device)
    h_prime = tem.index_add_(0, rows, edge_msg) # nd
    # h = h.float() 
    # h_prime = a @ h 
    # h_prime = h_prime.bfloat16()
    return h_prime

# Implementation of MPNN, which can become MLP or GCN depending on whether using message passing
class MPNN(nn.Module): 
    def __init__(self, in_channels, hidden_channels, out_channels, **kwargs):
        super(MPNN, self).__init__()
        self.config = PretrainedConfig()
        self.dropout = kwargs.get('dropout')# args.dropout
        self.num_layers = kwargs.get('num_layers')# args.num_layers
        self.ff_bias = True  # Use bias for FF layers in default

        self.bns = nn.BatchNorm1d(hidden_channels, affine=False, track_running_stats=False)
        self.activation = F.relu
        self.if_param = kwargs.get('if_param')

        if self.if_param: 
            self.fcs = nn.ModuleList([])
            self.fcs.append(nn.Linear(in_channels, hidden_channels, bias=self.ff_bias))
            for _ in range(self.num_layers - 2): self.fcs.append(nn.Linear(hidden_channels, hidden_channels, bias=self.ff_bias)) #1s
            self.fcs.append(nn.Linear(hidden_channels, out_channels, bias=self.ff_bias)) #1
            self.reset_parameters()
    

    def reset_parameters(self):
        for mlp in self.fcs: 
            nn.init.xavier_uniform_(mlp.weight, gain=1.414)
            nn.init.zeros_(mlp.bias)

    def forward(self, g, use_conv=True):
        
        x = g.graph_node
        edge_index = g.edge_index
        try:
            device = self.parameters().__next__().device
        except: 
            device = x.device
        x = x.to(device)
        edge_index = edge_index.to(device)
        for i in range(self.num_layers - 1):
            if self.if_param: x = x @ self.fcs[i].weight.t() 
            if use_conv: x = gcn_conv(x, edge_index)  # Optionally replace 'gcn_conv' with other conv functions in conv.py
            if self.ff_bias and self.if_param: x = x + self.fcs[i].bias
            try: 
                x = self.activation(self.bns(x))
            except: 
                x = self.activation((x))
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.if_param: x = x @ self.fcs[-1].weight.t() 
        if use_conv: x = gcn_conv(x, edge_index)
        if self.ff_bias and self.if_param: x = x + self.fcs[-1].bias
        return x

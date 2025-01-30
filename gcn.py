
"""
Reference: https://github.com/pmernyei/wiki-cs-dataset/blob/master/experiments/node_classification/gcn/gcn.py
"""

import torch.nn as nn
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(in_feats, n_hidden, add_self_loops=True))
        for i in range(n_layers - 1):
            self.layers.append(GCNConv(n_hidden, n_hidden, add_self_loops=True))
        self.layers.append(GCNConv(n_hidden, n_classes, add_self_loops=True))
        self.dropout = nn.Dropout(p=dropout)
        self.activation = activation
        
    def forward(self, x, edge_index):
        h = x
        for i, layer in enumerate(self.layers[:-1]):
            if i != 0:
                h = self.dropout(h)
            h = layer(h, edge_index)
            h = self.activation(h)
        h = self.layers[-1](h, edge_index)
        return h

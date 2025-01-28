import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_classes,
                 n_layers,
                 n_hidden,
                 dropout):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GCNConv(in_feats, n_hidden))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GCNConv(n_hidden, n_hidden))
        # output layer
        self.layers.append(GCNConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
        h = x
        for i, layer in enumerate(self.layers[:-1]):
            if i != 0:
                h = self.dropout(h)
            h = layer(h, edge_index)  # Pass x and edge_index to the GCNConv layer
        self.last_embeddings = h
        h = self.dropout(h)
        h = self.layers[-1](h, edge_index)  # Pass x and edge_index to the GCNConv layer
        return h


def accuracy(logits, labels, mask=None):
    if mask is not None:
        logits = logits[mask]
        labels = labels[mask]
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    acc = correct.item() * 1.0 / len(labels)
    return acc

def loss_scalar(logits, labels, mask, loss_fcn):
    if mask is not None:
        logits = logits[mask]
        labels = labels[mask]
    return loss_fcn(logits, labels).cpu().numpy().mean()

def evaluate(model, features, labels, mask, loss_fcn=None):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        acc = accuracy(logits, labels, mask)
        if loss_fcn is None:
            return acc
        else:
            return acc, loss_scalar(logits, labels, mask, loss_fcn)

def train_and_eval(data, stopping_patience, lr, weight_decay, device, n_layers=3, n_hidden=33, dropout=0.25):
    model = GCN(data, data.num_features, data.num_classes, n_layers, n_hidden, dropout).to(device)
    data = data.to(device)
    min_loss = 10000
    patience_left = stopping_patience
    epoch = 0
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr, weight_decay=weight_decay)

    while patience_left > 0:
        model.train()
        logits = model(data.x, data.edge_index)
        loss = loss_fcn(logits[data.train_mask], data.y[data.train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            eval_logits = model(data.x, data.edge_index)
            stopping_loss = loss_scalar(eval_logits, data.y, data.stopping_mask, loss_fcn)
            test_acc = accuracy(eval_logits, data.y, data.test_mask)

        if stopping_loss < min_loss:
            min_loss = stopping_loss
            patience_left = stopping_patience
        else:
            patience_left -= 1

        epoch += 1

    return test_acc

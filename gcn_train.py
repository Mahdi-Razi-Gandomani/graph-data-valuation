
"""
based on https://github.com/pmernyei/wiki-cs-dataset/blob/master/experiments/node_classification/train.py,
"""

import torch


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

def evaluate(model, features, edge_index, labels, mask, loss_fcn=None):
    model.eval()
    with torch.no_grad():
        logits = model(features, edge_index)
        acc = accuracy(logits, labels, mask)
        if loss_fcn is None:
            return acc
        else:
            return acc, loss_scalar(logits, labels, mask, loss_fcn)

def train_and_eval(data, model, stopping_patience, lr, weight_decay, device):
    model = model.to(device)
    data = data.to(device)

    min_loss = float('inf')
    patience_left = stopping_patience
    best_vars = None

    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

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

        if stopping_loss < min_loss:
            min_loss = stopping_loss
            patience_left = stopping_patience
            best_vars = {
                key: value.clone()
                for key, value in model.state_dict().items()
            }
        else:
            patience_left -= 1

    model.load_state_dict(best_vars)
    val_acc = evaluate(
        model, data.x, data.edge_index, data.y,
        data.val_mask
    )
    test_acc = evaluate(
        model, data.x, data.edge_index, data.y,
        data.test_mask
    )

    return val_acc, test_acc

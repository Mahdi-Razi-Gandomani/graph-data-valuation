
"""
This script implements the PC-Winter (Precedence-Constrained Winter) algorithm for graph data valuation.
Key features of the implementation include:

1. Evaluation Model: SGC (Simple Graph Convolution) is used, split into two parts:
   a) Local propagation: Implemented in generate_features_and_labels_ind function.
   b) Classifier training: Using an MLP (Multi-Layer Perceptron).

2. Local Propagation: generate_features_and_labels_ind function implements the local propagation strategy,
   combining previously propagated node features with partially propagated features of the target node.

3. Preorder Traversal: The main nested loops implement the preorder traversal of the contribution tree,
   a key component of the PC-Winter algorithm.

4. Hierarchical Truncation: Implemented to reduce computational complexity,
   truncating at both the 1-hop node level and the 2-hop node levels.

5. Inductive Setting: The script sets up an inductive learning environment by removing validation and test nodes
   from the training graph.

6. Value Accumulation: Throughout the traversal, values are accumulated for each node,
   representing their contribution to the model's performance.

The script uses command-line arguments to control algorithm behavior, including dataset selection,
number of hops, random seed, number of permutations, and truncation ratios.
"""

import torch
import torch.nn as nn
from torch_geometric.datasets import Planetoid, WikiCS
from torch_geometric.utils import k_hop_subgraph
import numpy as np
import pickle
import argparse
import torch.nn.functional as F
import copy
import torch_geometric.transforms as T
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.utils import spmm
from tqdm import tqdm
import os


class SGConvNoWeight(MessagePassing):
    
    """
    The modified SGConv operator without the trainable linear layer.
    This class implements the feature propagation mechanism used in the local propagation strategy.
    """
    
    def __init__(self, K: int = 2,
                 cached: bool = False, add_self_loops: bool = True,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.K = K
        self.cached = cached
        self.add_self_loops = add_self_loops

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        if isinstance(edge_index, Tensor):
            edge_index, edge_weight = gcn_norm(
                edge_index, edge_weight, x.size(self.node_dim), False,
                self.add_self_loops, self.flow, dtype=x.dtype)
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, x.size(self.node_dim), False,
                self.add_self_loops, self.flow, dtype=x.dtype)

        for k in range(self.K):
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                               size=None)
        return x

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.channels}, K={self.K})')


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, n_hidden_layers, hidden_dim, dropout):
        super(MLP, self).__init__()
        self.layers = nn.Sequential()
        if n_hidden_layers>0:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            for i in range(n_hidden_layers - 1):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.layers.append(nn.ReLU())
                if dropout > 0:
                    self.layers.append(nn.Dropout(p=dropout))
            self.layers.append(nn.Linear(hidden_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, output_dim))

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.layers(x)
        return F.log_softmax(out, dim=1)

    def predict(self, x):
        output = self.forward(x)
        return output

    def fit(self, X, y, val_X, val_y, num_iter, lr, weight_decay, patience):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        best_val_acc = 0
        patience_left = patience

        for epoch in range(num_iter):
            self.train()
            optimizer.zero_grad()
            output = self(X)
            loss = F.nll_loss(output, y)
            loss.backward()
            optimizer.step()

            self.eval()
            with torch.no_grad():
                val_pred = self(val_X).argmax(dim=1)
                val_acc = (val_pred == val_y).float().mean().item()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_left = patience
            else:
                patience_left -= 1

                if patience_left <= 0:
                    break

        return best_val_acc

            
def adjacency_to_edge_list(adj_matrix):
    """
    Convert an adjacency matrix to an edge list representation.
    """
    # Convert the adjacency matrix to COO format
    source, target = adj_matrix.nonzero().unbind(dim=1)
    # Create the edge_index tensor
    edge_index = torch.stack([source, target], dim=0)
    return edge_index

def stack_torch_tensors(input_tensors):
    unrolled = [input_tensors[k].view(-1,1) for k in range(len(input_tensors))]
    return torch.cat(unrolled)

def generate_features_and_labels_ind(cur_hop_1_list, cur_hop_2_list, cur_labeled_list, target_node, node_map, ind_edge_index, data, device):

    """
    This function implements the local propagation strategy for the PC-Winter algorithm.
    It generates features and labels for the inductive setting, considering the current state
    of the graph during a permutation.

    The key step is the concatenation of previously propagated features with the
    partially propagated features of the current target node:
    train_features = torch.cat((X_ind_propogated[cur_labeled_list], X_ind_propogated_temp_[target_node].unsqueeze(0)), dim=0)
    train_labels = torch.cat((data.y[cur_labeled_list], data.y[target_node].unsqueeze(0)), dim=0)

    This approach allows for efficient computation of node contributions in the PC-Winter algorithm.
    """
    
    A_temp = torch.zeros((data.x.size(0), data.x.size(0)), device=device)
    A_temp[ind_edge_index[0], ind_edge_index[1]] = 1
    
    mask = torch.zeros_like(A_temp)
    mask[target_node, cur_hop_1_list] = 1
    mask[cur_hop_1_list, target_node] = 1
    if len(cur_hop_1_list) > 1:
        for hop_1_node in cur_hop_1_list[:-1]:
            hop_2_list = list(node_map[target_node][hop_1_node].keys())
            mask[hop_1_node, hop_2_list] = 1
            mask[hop_2_list, hop_1_node] = 1
        mask[cur_hop_1_list[-1], cur_hop_2_list] = 1
        mask[cur_hop_2_list, cur_hop_1_list[-1]] = 1
    else:
        mask[cur_hop_1_list[-1], cur_hop_2_list] = 1
        mask[cur_hop_2_list, cur_hop_1_list[-1]] = 1
    
    conv = SGConvNoWeight(K=2)
    cur_edge_index = adjacency_to_edge_list(mask)
    X_ind_propogated_temp_ = conv(data.x, cur_edge_index)
        
    train_features = torch.cat((X_ind_propogated[cur_labeled_list], X_ind_propogated_temp_[target_node].unsqueeze(0)), dim=0)
    train_labels = torch.cat((data.y[cur_labeled_list], data.y[target_node].unsqueeze(0)), dim=0)

    return train_features, train_labels

def evaluate_retrain_model(model_class, num_features, num_classes, n_hidden_layers, hidden_dim, dropout, 
                           train_features, train_labels, val_features, val_labels, device, num_iter, lr, weight_decay, patience):
    
    """
    This function creates, trains, and evaluates a model on the given data.
    It's used to compute the utility function in the PC-Winter algorithm.
    The utility is measured as the validation accuracy of the trained model.
    """
    
    # Create and train the model
    model = model_class(num_features, num_classes, n_hidden_layers, hidden_dim, dropout).to(device)
    val_acc = model.fit(train_features, train_labels, val_features, val_labels, num_iter, lr, weight_decay, patience)
    return val_acc

def generate_maps(train_idx_list, edge_index):

    """
    This function generates the necessary data structures for efficient computation
    of the PC-Winter algorithm, including the labeled_to_player_map which represents
    the contribution tree structure. 
    
    The key chain stands for one contribution path of a node in a computational tree:
    [labeled][labeled][labeled] is a labeled node;
    [labeled][hop_1_node][hop_1_node] is a label' node's 1-distance neighbor;
    [labeled][hop_1_node][hop_2_node] is a label' node's 2-distance neighbor;
    Here the key index is the node index in the graph. 
    """

    labeled_to_player_map = {}
    sample_value_dict = {}
    sample_counter_dict = {}
    for labeled in train_idx_list:
        hop_1_nodes, _,_,_ = k_hop_subgraph(int(labeled), num_hops=1, edge_index=edge_index, relabel_nodes=False)
        hop_1_nodes_list = list(hop_1_nodes.cpu().numpy())
        hop_1_nodes_list.remove(labeled) 
        labeled_to_player_map[labeled] = {}
        sample_value_dict[labeled] = {}
        sample_counter_dict[labeled] = {}
        labeled_to_player_map[labeled][labeled] = {}
        sample_value_dict[labeled][labeled] = {}
        sample_counter_dict[labeled][labeled] = {}
        
        for hop_1_node in hop_1_nodes_list:
            sub_nodes_2, _,_,_ = k_hop_subgraph(int(hop_1_node), num_hops=1, edge_index=edge_index, relabel_nodes=False)
            sub_nodes_2_list = list(sub_nodes_2.cpu().numpy())
            sub_nodes_2_list.remove(hop_1_node) 
            labeled_to_player_map[labeled][hop_1_node] = {}
            sample_value_dict[labeled][hop_1_node] = {}
            sample_counter_dict[labeled][hop_1_node] = {}
                
            for hop_2_node in sub_nodes_2_list:
                labeled_to_player_map[labeled][hop_1_node][hop_2_node] = [hop_2_node]
                sample_value_dict[labeled][hop_1_node][hop_2_node] = 0
                sample_counter_dict[labeled][hop_1_node][hop_2_node] = 0
            labeled_to_player_map[labeled][hop_1_node][hop_1_node] = [hop_1_node]
            sample_value_dict[labeled][hop_1_node][hop_1_node] = 0
            sample_counter_dict[labeled][hop_1_node][hop_1_node] = 0
            
        labeled_to_player_map[labeled][labeled][labeled] = [labeled]
        sample_value_dict[labeled][labeled][labeled] = 0
        sample_counter_dict[labeled][labeled][labeled] = 0

    return labeled_to_player_map, sample_value_dict, sample_counter_dict

def get_subgraph_data(data_edge_index, mask):
    
    """
    This function extracts a subgraph from the given graph based on a mask.
    The resulting subgraph only contains edges between nodes in the mask.
    """
    
    # Nodes to be considered
    edge_index = data_edge_index.clone()
    nodes = mask.nonzero().reshape(-1)

    # Extract the edges for these nodes
    edge_mask_src = (edge_index[0].unsqueeze(-1) == nodes.unsqueeze(0)).any(dim=-1)
    edge_mask_dst = (edge_index[1].unsqueeze(-1) == nodes.unsqueeze(0)).any(dim=-1)
    edge_mask = edge_mask_src & edge_mask_dst

    sub_edge_index = edge_index[:, edge_mask]
    return sub_edge_index
      
def propagate_features(edge_index, node_features):
    """SGC propagation of node features using the given edge_index."""
    A = torch.zeros((node_features.size(0), node_features.size(0)), device=device)
    A[edge_index[0], edge_index[1]] = 1
    A_hat = A + torch.eye(A.size(0), device=device)
    D_hat_diag = A_hat.sum(dim=1).pow(-0.5)
    D_hat = torch.diag(D_hat_diag)
    L_norm = D_hat.mm(A_hat).mm(D_hat)
    return L_norm.mm(L_norm.mm(node_features))

def parse_args():
    parser = argparse.ArgumentParser(description="Network")
    parser.add_argument('--dataset', default='Cora', help='Input dataset.')
    # parser.add_argument('--num_hops', type=int, default=2, help='Number of hops.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for permutation.')
    parser.add_argument('--num_perm', type=int, default=10, help='Number of permutations.')
    # parser.add_argument('--label_trunc_ratio', type=float, default=0, help='Label trunc ratio')
    parser.add_argument('--group_trunc_ratio_hop_1', type=float, default=0.5, help='Hop 1 Group trunc ratio')
    parser.add_argument('--group_trunc_ratio_hop_2', type=float, default=0.7, help='Hop 2 Group trunc ratio.')
    parser.add_argument( '--verbose', type = bool, default = True)
    return parser.parse_args()


if __name__ == "__main__":    
    # Parse command line arguments
    args = parse_args()
    print(args)

    # Set up dataset and model parameters
    dataset_name = args.dataset
    seed = args.seed
    num_perm = args.num_perm
    group_trunc_ratio_hop_1 = args.group_trunc_ratio_hop_1
    group_trunc_ratio_hop_2 = args.group_trunc_ratio_hop_2
    verbose = args.verbose

    # Hyperparameters
    num_epochs = 200
    lr = 0.003
    weight_decay = 5e-4
    n_hidden_layers = 1
    hidden_dim =  35
    patience = 100
    dropout = 0.25
    
    np.random.seed(seed)
    
    # Load dataset
    dataset_loaders = {
        'WikiCS': lambda: WikiCS(root='dataset/WikiCS', transform=T.NormalizeFeatures()),
        'Cora': lambda: Planetoid(root='dataset/Cora', name='Cora', transform=T.NormalizeFeatures()),
        'Citeseer': lambda: Planetoid(root='dataset/Citeseer', name='Citeseer', transform=T.NormalizeFeatures()),
        'Pubmed': lambda: Planetoid(root='dataset/Pubmed', name='Pubmed', transform=T.NormalizeFeatures()),
    }

    if dataset_name not in dataset_loaders:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets are {list(dataset_loaders.keys())}")

    dataset = dataset_loaders[dataset_name]()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dataset[0].to(device)
           
    if dataset_name == 'WikiCS':
        split_idx = 0
        data.train_mask = data.train_mask[:, split_idx]
        data.val_mask = data.val_mask[:, split_idx] 
        data.stopping_mask = data.stopping_mask[:, split_idx]
    
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask
       
    if verbose:
        train_size = train_mask.sum().item()
        val_size = val_mask.sum().item()
        test_size = test_mask.sum().item()
        print(f'Dataset: {dataset_name}')
        print(f"Train Size: {train_size}")
        print(f"Validation Size: {val_size}")
        print(f"Test Size: {test_size}")

    # Prepare validation and test data
    val_edge_index =  get_subgraph_data(data.edge_index, val_mask)
    X_val_propogated = propagate_features(val_edge_index , data.x)
    test_edge_index =  get_subgraph_data(data.edge_index, test_mask)
    X_test_propogated = propagate_features(test_edge_index, data.x)
    
    val_features = X_val_propogated[val_mask]
    val_labels = data.y[val_mask]
    test_features = X_test_propogated[test_mask]
    test_labels = data.y[test_mask]

    # Create inductive edge index (removing edges to val/test nodes)
    inductive_edge_index = []
    for src, tgt in data.edge_index.t().tolist():
        if not (val_mask[src] or test_mask[src] or val_mask[tgt] or test_mask[tgt]):
            inductive_edge_index.append([src, tgt])
    inductive_edge_index = torch.tensor(inductive_edge_index).t().contiguous()
    X_ind_propogated = propagate_features(inductive_edge_index, data.x)
    
    if verbose:
        original_edge_count = data.edge_index.size(1)  
        inductive_edge_count = inductive_edge_index.size(1)  
        print(f"Original Edge Count: {original_edge_count}")
        print(f"Inductive Edge Count: {inductive_edge_count}")
    
    # Prepare storage data structures for PC-Winter algorithm
    train_idx = torch.nonzero(train_mask).cpu().numpy().flatten()
    labeled_node_list = list(train_idx)
    labeled_to_player_map, sample_value_dict, sample_counter_dict = \
            generate_maps( list(train_idx), inductive_edge_index)
    
    #Store the performance of different seed, permutation index, added new contribution path and accrued performace
    perf_dict = {
        'dataset': [], 'seed': [], 'perm': [], 'label': [],
        'first_hop': [], 'second_hop': [], 'accu': []
    }
    
    # Main loop for PC-Winter algorithm with online Pre-order traversal
    for i in range(num_perm):
        np.random.shuffle(labeled_node_list) # Randomize order of labeled nodes
        cur_labeled_node_list = [] 
        pre_performance = 0
        
        for labeled_node in tqdm(labeled_node_list, desc=f'Perm {i + 1}/{num_perm}', unit='node'):
            sample_value_dict_copy = copy.deepcopy(sample_value_dict)
            # Process 1-hop neighbors
            cur_hop_1_list = []
            hop_1_list = list(labeled_to_player_map [labeled_node].keys())
            np.random.shuffle(hop_1_list) # Randomize order
            # Keep the precedence constraint between labeled and 1-hop neighbor by putting labeled node front
            hop_1_list.remove(labeled_node)
            hop_1_list.insert(0, labeled_node)
            # print ('hop_1_list before truncation', len(hop_1_list), 'group_trunc_ratio_hop_1:', group_trunc_ratio_hop_1)
            truncate_length = int(np.ceil((len(hop_1_list) - 1) * (1- group_trunc_ratio_hop_1))) + 1
            truncate_length = min(truncate_length, len(hop_1_list))
            hop_1_list = hop_1_list[:truncate_length]
            # print ('hop_1_list after truncation', len(hop_1_list) )
            
            # print ('labeled_node iteration:', i)
            # print ('current target labeled_node:',  cur_labeled_node_list, '=>', labeled_node)
            # print ('hop_1_list ', hop_1_list )
            
            for player_hop_1 in hop_1_list:
                # Process 2-hop neighbors
                cur_hop_2_list = []
                cur_hop_1_list += [player_hop_1]
                hop_2_list = list(labeled_to_player_map [labeled_node][player_hop_1].keys())
                np.random.shuffle(hop_2_list) # Randomize order
                # keep the precedence constraint between 1-hop neighbor and 2-hop neighbor 
                hop_2_list.remove(player_hop_1)
                hop_2_list.insert(0, player_hop_1)
                # print ('hop_2_list before truncation', len(hop_2_list), 'group_trunc_ratio_hop_2:', group_trunc_ratio_hop_2)
                truncate_length = int(np.ceil((len(hop_2_list) - 1) * (1-group_trunc_ratio_hop_2))) + 1
                truncate_length = min(truncate_length, len(hop_2_list))
                hop_2_list = hop_2_list[:truncate_length]
                # print ('hop_2_list after truncation', len(hop_2_list) )
                
                for player_hop_2 in hop_2_list:
                    cur_hop_2_list += [player_hop_2]
                    # Local propagation and performance computation
                    ind_train_features, ind_train_labels = generate_features_and_labels_ind(cur_hop_1_list, cur_hop_2_list, cur_labeled_node_list,
                                                labeled_node, labeled_to_player_map, inductive_edge_index, data, device)
                    val_acc = evaluate_retrain_model(MLP, dataset.num_features, dataset.num_classes, n_hidden_layers, hidden_dim, dropout,
                                                     ind_train_features, ind_train_labels, val_features, val_labels, 
                                                     device, num_epochs, lr, weight_decay, patience)
                    # calculate marginal contribution 
                    sample_value_dict[labeled_node][player_hop_1][player_hop_2] += ( val_acc - pre_performance)
                    sample_counter_dict [labeled_node][player_hop_1][player_hop_2] += 1
                    pre_performance = val_acc

                    # Record performance data
                    perf_dict['dataset'] += [dataset_name]
                    perf_dict['seed'] += [seed]
                    perf_dict['perm'] += [i]
                    perf_dict['label'] += [labeled_node]
                    perf_dict['first_hop'] += [player_hop_1]
                    perf_dict['second_hop'] += [player_hop_2]
                    perf_dict['accu'] += [val_acc]         
                    
                    # print('pre_performance ',pre_performance  ,'val_acc', val_acc)

            # Update labeled node list and compute full group accuracy
            cur_labeled_node_list += [labeled_node]
            ind_train_features = X_ind_propogated[ cur_labeled_node_list]
            ind_train_labels = data.y[cur_labeled_node_list]
            val_acc = evaluate_retrain_model(MLP, dataset.num_features, dataset.num_classes, n_hidden_layers, hidden_dim, dropout,
                                        ind_train_features, ind_train_labels, val_features, val_labels,
                                        device, num_epochs, lr, weight_decay, patience)
            pre_performance = val_acc
            # print('full group acc:', val_acc)
        
    path = 'value/'
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, f'{dataset_name}_{seed}_{num_perm}_{group_trunc_ratio_hop_1}_{group_trunc_ratio_hop_2}_pc_value.pkl'), "wb") as f:
        pickle.dump(sample_value_dict, f) 

    with open(os.path.join(path, f'{dataset_name}_{seed}_{num_perm}_{group_trunc_ratio_hop_1}_{group_trunc_ratio_hop_2}_pc_value_count.pkl'), "wb") as f:
        pickle.dump(sample_counter_dict, f)

    with open(os.path.join(path, f'{dataset_name}_{seed}_{num_perm}_{group_trunc_ratio_hop_1}_{group_trunc_ratio_hop_2}_perf.pkl'), "wb") as f:
        pickle.dump(perf_dict, f)

    print('\nDone!')

import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros

import numpy as np
# from core.encoders import *
import json
from torch import optim

from cortex_DIM.nn_modules.mi_networks import MIFCNet, MI1x1ConvNet
from losses import *

num_atom_type = 120 #including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3 

class GlobalDiscriminator(nn.Module):
    def __init__(self, args, input_dim):
        super().__init__()
        
        self.l0 = nn.Linear(32, 32)
        self.l1 = nn.Linear(32, 32)

        self.l2 = nn.Linear(512, 1)
    def forward(self, y, M, data):

        adj = Variable(data['adj'].float(), requires_grad=False).cuda()
        # h0 = Variable(data['feats'].float()).cuda()
        batch_num_nodes = data['num_nodes'].int().numpy()
        M, _ = self.encoder(M, adj, batch_num_nodes)
        # h = F.relu(self.c0(M))
        # h = self.c1(h)
        # h = h.view(y.shape[0], -1)
        h = torch.cat((y, M), dim=1)
        h = F.relu(self.l0(h))
        h = F.relu(self.l1(h))
        return self.l2(h)

class PriorDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.l0 = nn.Linear(input_dim, input_dim)
        self.l1 = nn.Linear(input_dim, input_dim)
        self.l2 = nn.Linear(input_dim, 1)

    def forward(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return torch.sigmoid(self.l2(h))

class FF(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # self.c0 = nn.Conv1d(input_dim, 512, kernel_size=1)
        # self.c1 = nn.Conv1d(512, 512, kernel_size=1)
        # self.c2 = nn.Conv1d(512, 1, kernel_size=1)
        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, input_dim)
        # self.c0 = nn.Conv1d(input_dim, 512, kernel_size=1, stride=1, padding=0)
        # self.c1 = nn.Conv1d(512, 512, kernel_size=1, stride=1, padding=0)
        # self.c2 = nn.Conv1d(512, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)
    

class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not. 
        

    See https://arxiv.org/abs/1810.00826
    """
    def __init__(self, emb_dim, aggr = "add"):
        super(GINConv, self).__init__()
        #multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        # return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings, norm = norm)
        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GCNConv(MessagePassing):

    def __init__(self, emb_dim, aggr = "add"):
        super(GCNConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        norm = self.norm(edge_index, x.size(0), x.dtype)

        x = self.linear(x)

        return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings, norm = norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)


class GATConv(MessagePassing):
    def __init__(self, emb_dim, heads=2, negative_slope=0.2, aggr = "add"):
        super(GATConv, self).__init__()

        self.aggr = aggr

        self.emb_dim = emb_dim
        self.heads = heads
        self.negative_slope = negative_slope

        self.weight_linear = torch.nn.Linear(emb_dim, heads * emb_dim)
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))

        self.bias = torch.nn.Parameter(torch.Tensor(emb_dim))

        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, heads * emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, heads * emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):

        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        x = self.weight_linear(x).view(-1, self.heads, self.emb_dim)
        return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, edge_index, x_i, x_j, edge_attr):
        edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_j += edge_attr

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        aggr_out = aggr_out.mean(dim=1)
        aggr_out = aggr_out + self.bias

        return aggr_out


class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim, aggr = "mean"):
        super(GraphSAGEConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        x = self.linear(x)

        return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p = 2, dim = -1)



class GNN(torch.nn.Module):
    """
    

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """
    def __init__(self, num_layer, emb_dim, JK = "last", drop_ratio = 0, gnn_type = "gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr = "add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    #def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        old_x = x
        x = self.x_embedding1(old_x[:,0]) + self.x_embedding2(old_x[:,1])
        # print("self.x_embedding1(x[:,0]): {}".format(self.x_embedding1(old_x[:,0])))
        # print("self.x_embedding2(x[:,0]): {}".format(self.x_embedding2(old_x[:,1])))

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            #h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]

        return node_representation

class GNN_Virtualnode(torch.nn.Module):
    """


    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """
    def __init__(self, num_layer, emb_dim, JK = "last", drop_ratio = 0, gnn_type = "gin"):
        super(GNN_Virtualnode, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        ### set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr = "add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        ### List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()

        for layer in range(num_layer - 1):
            self.mlp_virtualnode_list.append(torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), \
                torch.nn.Linear(2*emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU()))

        self.linear_pred_atoms = torch.nn.Linear(emb_dim, 121)
        self.linear_pred_bonds = torch.nn.Linear(emb_dim, 6)

    #def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        elif len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])

        ### virtual node embeddings for graphs
        virtualnode_embedding = self.virtualnode_embedding(torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))

        h_list = [x]
        for layer in range(self.num_layer):
            ### add message from virtual nodes to graph nodes
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]

            ### Message passing among graph nodes
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            #h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)

            ### update the virtual nodes
            if layer < self.num_layer - 1:
                ### add message from graph nodes to virtual nodes
                virtualnode_embedding_temp = global_add_pool(h_list[layer], batch) + virtualnode_embedding
                ### transform virtual nodes using MLP

                virtualnode_embedding = F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio, training = self.training)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]

        return node_representation


class GNN_graphpred(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat
        
    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """
    def __init__(self, num_layer, emb_dim, num_tasks, JK = "last", drop_ratio = 0, graph_pooling = "mean", gnn_type = "gin"):
        super(GNN_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type = gnn_type)

        #Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        #For graph-level binary classification
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1
        
        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(self.mult * (self.num_layer + 1) * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.mult * self.emb_dim, self.num_tasks)

    def from_pretrained(self, model_file):
        #self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        self.gnn.load_state_dict(torch.load(model_file))

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.gnn(x, edge_index, edge_attr)

        return self.graph_pred_linear(self.pool(node_representation, batch))


class GNN_graphpred_Virtualnode(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat

    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """
    def __init__(self, num_layer, emb_dim, num_tasks, JK = "last", drop_ratio = 0, graph_pooling = "mean", gnn_type = "gin"):
        super(GNN_graphpred_Virtualnode, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN_Virtualnode(num_layer, emb_dim, JK, drop_ratio, gnn_type = gnn_type)


        #Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        #For graph-level binary classification
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1

        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(self.mult * (self.num_layer + 1) * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.mult * self.emb_dim, self.num_tasks)

        ### set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        ### List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()

        for layer in range(num_layer - 1):
            self.mlp_virtualnode_list.append(torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), \
                torch.nn.Linear(2*emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU()))

    def from_pretrained(self, model_file):
        #self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        self.gnn.load_state_dict(torch.load(model_file))

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.gnn(x, edge_index, edge_attr, batch)
        ### virtual node embeddings for graphs
        virtualnode_embedding = self.virtualnode_embedding(torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))

        # return self.graph_pred_linear(self.pool(node_representation, batch) + virtualnode_embedding)
        return self.gnn.linear_pred_atoms(self.pool(node_representation, batch) + virtualnode_embedding)

        # return self.graph_pred_linear(self.)

if __name__ == "__main__":
    pass

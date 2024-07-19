import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv

class GraphConvolution(nn.Module):
    """图卷积层"""
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

# class Attention(nn.Module):
#     def __init__(self, num_nodes):
#         super(Attention, self).__init__()
#         self.num_nodes = num_nodes
#         self.attn = nn.Linear(num_nodes * num_nodes * 2, 1)  # 注意这里的输入维度

#     def forward(self, adj, LS_adj):

#         adj_flat = adj.to_dense().flatten()
#         LS_adj_flat = LS_adj.to_dense().flatten()
#         x = torch.cat((adj_flat, LS_adj_flat), dim=0)
#         alpha = torch.sigmoid(self.attn(x))
       
#         return alpha, 1 - alpha


# class LS_GCN(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout, num_nodes):
#         super(LS_GCN, self).__init__()

#         self.gc1 = GraphConvolution(nfeat, nhid)
#         self.gc2 = GraphConvolution(nhid, nclass)
#         self.dropout = dropout

#         # Attention layer
#         self.attn = Attention(num_nodes = num_nodes)

#     def forward(self, x, adj, LS_adj):
#         alpha, beta = self.attn(adj, LS_adj)
#         print(alpha,beta)
#         combined_adj = alpha * adj + beta * LS_adj

#         x = F.relu(self.gc1(x, combined_adj))
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.gc2(x, combined_adj)
#         return F.log_softmax(x, dim=1)

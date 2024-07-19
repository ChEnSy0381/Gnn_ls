import numpy as np
import scipy.sparse as sp
import torch
from LS_MARTIX import LS_matrix
from LS_corr_matrix import LS_corr_matrix
import networkx as nx
from sklearn.model_selection import train_test_split

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_data(path, dataset,alpha):
    print('Loading {} dataset...'.format(dataset))
    if alpha is None:
        alpha = 1
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = np.array(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.str)
    idx_map = {j: i for i, j in enumerate(idx)}
    # #cora
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    #citeseeer 需要预处理，存在节点在cits中而不在content中
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.str)
    edges_mapped = []
    for edge in edges_unordered:
        if edge[0] in idx_map and edge[1] in idx_map:
            edges_mapped.append([idx_map[edge[0]], idx_map[edge[1]]])
        else:
            print(f'Warning: skipping edge with missing node {edge}')
    edges = np.array(edges_mapped, dtype=np.int32)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    # Load LS algorithm generated graph
    G = nx.from_scipy_sparse_matrix(adj)
    new_D, LS_mat = LS_matrix(G)  # 调用LS算法

    # correlation_matrix = np.corrcoef(features)
    # new_D, LS_mat = LS_corr_matrix(G,correlation_matrix)  # 调用LS_corr算法

    #  原矩阵为节点指向最大度邻居，local leader指向local leader，现将其转置
    # # 转置矩阵
    # LS_mat = np.transpose(LS_mat)

    LS_adj = sp.coo_matrix(LS_mat)
    # 忽略方向性
    LS_adj = LS_adj + LS_adj.T.multiply(LS_adj.T > LS_adj) - LS_adj.multiply(LS_adj.T > LS_adj)

    # Combine the original adjacency matrix with the LS adjacency matrix
    beta = 1 - alpha
    adj = alpha * adj + beta * LS_adj

    features = normalize(features)
    adj = normalize(adj)

    # 首先将所有节点按类别分开
    class_labels = np.argmax(labels, axis=1)
    class_indices = [np.where(class_labels == i)[0] for i in range(labels.shape[1])]

    # 将每个类别的节点划分为10%训练集、20%验证集和70%测试集
    idx_train, idx_val, idx_test = [], [], []
    for indices in class_indices:
        idx_train_part, idx_test_part = train_test_split(indices, test_size=0.7, random_state=42)
        idx_train_part, idx_val_part = train_test_split(idx_train_part, test_size=0.3333, random_state=42)  # 1/3 of 30% is 10%

        idx_train.extend(idx_train_part)
        idx_val.extend(idx_val_part)
        idx_test.extend(idx_test_part)

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(np.where(labels)[1])
    adj = torch.FloatTensor(np.array(adj.todense()))

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

def ls_load_data(path, dataset):
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.str)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.str)
    edges_mapped = []
    for edge in edges_unordered:
        if edge[0] in idx_map and edge[1] in idx_map:
            edges_mapped.append([idx_map[edge[0]], idx_map[edge[1]]])
        else:
            print(f'Warning: skipping edge with missing node {edge}')
    edges = np.array(edges_mapped, dtype=np.int32)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # Load LS algorithm generated graph
    G = nx.from_scipy_sparse_matrix(adj)
    new_D, LS_mat = LS_matrix(G,seed=1)  # 这里调用我的LS算法，假设返回一个有向图的邻接矩阵
    LS_adj = sp.coo_matrix(LS_mat)
    #  原矩阵为节点指向最大度邻居，local leader指向local leader，现将其转置
    # # 转置矩阵
    # LS_mat = np.transpose(LS_mat)
    #对称处理
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    LS_adj = LS_adj + LS_adj.T.multiply(LS_adj.T > LS_adj) - LS_adj.multiply(LS_adj.T > LS_adj)
    
    # Normalize
    adj = normalize(adj + sp.eye(adj.shape[0]))
    LS_adj = normalize(LS_adj + sp.eye(LS_adj.shape[0]))

    # 首先将所有节点按类别分开
    class_labels = np.argmax(labels, axis=1)
    class_indices = [np.where(class_labels == i)[0] for i in range(labels.shape[1])]

    # 将每个类别的节点划分为10%训练集、20%验证集和70%测试集
    idx_train, idx_val, idx_test = [], [], []
    for indices in class_indices:
        idx_train_part, idx_test_part = train_test_split(indices, test_size=0.7, random_state=42)
        idx_train_part, idx_val_part = train_test_split(idx_train_part, test_size=0.3333, random_state=42)  # 1/3 of 30% is 10%

        idx_train.extend(idx_train_part)
        idx_val.extend(idx_val_part)
        idx_test.extend(idx_test_part)


    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    adj = sparse_mx_to_torch_sparse_tensor(adj)
    LS_adj = sparse_mx_to_torch_sparse_tensor(LS_adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, LS_adj, features, labels, idx_train, idx_val, idx_test


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

# -*- coding: utf-8 -*-

import numpy as np

import torch
from torch_geometric.utils import negative_sampling

import scipy.sparse as sp
from sklearn.metrics import f1_score, accuracy_score


def csr_to_sparse_tensor(csr_matrix: sp.csr_matrix):
    """Convert a scipy sparse matrix to a torch sparse tensor."""

    coo_matrix = csr_matrix.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((coo_matrix.row, coo_matrix.col)).astype(np.int64))
    values = torch.from_numpy(coo_matrix.data)
    shape = torch.Size(coo_matrix.shape)

    return torch.sparse.FloatTensor(indices, values, shape)


def sparse_tensor_to_csr(torch_sparse: torch.sparse.FloatTensor):
    """Convert a torch sparse tensor to a scipy sparse matrix."""

    m_index = torch_sparse._indices().numpy()
    sp_matrix = sp.coo_matrix(
        (torch_sparse._values().numpy(), (m_index[0], m_index[1])),
        shape=(torch_sparse.size()[0], torch_sparse.size()[1]))
    return sp_matrix.tocsr()


class Normalize:
    def __init__(self, data_range):
        if data_range != None:
            self.a, self.b = data_range
        else:
            self.a, self.b = None, None

    def __call__(self, data: torch.FloatTensor):
        if self.a != None and self.b != None:
            data_min = data.min(dim=0)[0]
            data_max = data.max(dim=0)[0]
            delta = data_max - data_min
            delta_zero_index = torch.nonzero(delta==0, as_tuple=False).squeeze()
            delta_zero_value = data[:, delta_zero_index]
            data = (data - data_min) * (self.b - self.a) / delta + self.a
            # process the features with delta = 0
            data[:, delta_zero_index] = delta_zero_value
        return data


def normalize_attributes(attr_matrix: torch.FloatTensor):
    epsilon = 1e-12
    if isinstance(attr_matrix, sp.csr_matrix):
        attr_norms = sp.linalg.norm(attr_matrix, ord=1, axis=1)
        attr_invnorms = 1 / np.maximum(attr_norms, epsilon)
        attr_mat_norm = attr_matrix.multiply(attr_invnorms[:, np.newaxis])
    else:
        attr_norms = np.linalg.norm(attr_matrix, ord=1, axis=1)
        attr_invnorms = 1 / np.maximum(attr_norms, epsilon)
        attr_mat_norm = attr_matrix * attr_invnorms[:, np.newaxis]
    return attr_mat_norm


def multiclass_f1(labels: torch.LongTensor, preds: torch.FloatTensor):
    preds = torch.argmax(preds, dim=1)
    return f1_score(labels, preds, average='micro')


def multilabel_f1(labels: torch.FloatTensor, preds: torch.FloatTensor):
    preds[preds > 0] = 1
    preds[preds <= 0] = 0
    return f1_score(labels, preds, average="micro")

def accuracy(labels: torch.LongTensor, preds: torch.LongTensor):
    return accuracy_score(labels, preds)


def get_pos_neg_edges(split_edge: dict, mode: str, num_nodes: int = None):
    pos_edge = split_edge[mode]['edge']
    # negative sample for training
    if mode == 'train':
        neg_edge = negative_sampling(
            split_edge['train']['edge'].T, num_nodes, pos_edge.size(0)).reshape(-1, 2)
    # valid and test
    else:
        neg_edge = split_edge[mode]['edge_neg']
    return pos_edge, neg_edge


class AUCLoss:
    def __call__(self, pos_out: torch.FloatTensor, neg_out: torch.FloatTensor):
        pos_out = pos_out.reshape(-1, 1)
        neg_out = neg_out.reshape(-1, 1)
        return torch.square(1 - (pos_out - neg_out)).sum()

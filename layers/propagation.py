# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sp

import torch

from utils import csr_to_sparse_tensor


def calc_A_hat(adj: sp.csr_matrix, add_self_loops: bool = True):
    # number of nodes
    n = adj.shape[0]
    if add_self_loops:
        # add self-loop
        adj = adj + sp.eye(n)
    # return self as a flattened ndarray
    D_vector = np.sum(adj, axis=1).A1
    # W = D^{-1/2}AD^{-1/2}
    D_vector_inverse = 1 / np.sqrt(D_vector)
    D_vector_inverse[D_vector_inverse == np.inf] = 0.
    D_inverse = sp.diags(D_vector_inverse)
    return D_inverse @ adj @ D_inverse


def calc_pagerank_exact(adj: sp.csr_matrix, x: torch.FloatTensor, alpha: float):
    n = adj.shape[0]
    A_hat = calc_A_hat(adj)
    A_inner = sp.eye(n) - (1 - alpha) * A_hat
    pagerank_matrix = torch.FloatTensor(alpha * np.linalg.inv(A_inner.toarray()))
    return pagerank_matrix @ x


def calc_pagerank_iters(adj: sp.csr_matrix,
                        x: torch.FloatTensor,
                        alpha: float, num_iters: int):
    A_hat = calc_A_hat(adj)
    W = csr_to_sparse_tensor((1 - alpha) * A_hat)
    middle_features = x
    for _ in range(num_iters):
        middle_features = W @ middle_features + alpha * x
    return middle_features


class KProp:
    def __init__(self, steps: int, normalize: bool = True):
        self.k = steps
        self.normalize = normalize

    def __call__(self, x: torch.FloatTensor, adj: sp.csr_matrix):
        if self.k <= 0:
            return x
        if self.normalize:
            adj = csr_to_sparse_tensor(calc_A_hat(adj, add_self_loops=False))
        for _ in range(self.k):
            x = adj @ x
        return x

# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sp

import torch
import torch.nn.functional as F
from perturbation import FeaturePerturbation

from collections import defaultdict
tree = lambda: defaultdict(tree)


class Graph:
    def __init__(self,
                adj: np.array = None,
                attr: torch.FloatTensor = None):
        self.adj = adj
        self.attr = attr

    def load_node_dataset(dataset_str: str):
        init_data = tree()
        dataset_path = './datasets/' + dataset_str + '/'
        init_data['adj'] = sp.load_npz(dataset_path+dataset_str+'.npz')
        init_data['attr'] = torch.FloatTensor(
            np.load(dataset_path + dataset_str + '_feat.npy'))
        return Graph(**init_data)

    def load_link_dataset(dataset_str: str):
        init_data = tree()
        dataset_path = './datasets/' + dataset_str + '/'
        init_data['adj'] = sp.load_npz(dataset_path+dataset_str+'_link.npz')
        init_data['attr'] = torch.FloatTensor(
            np.load(dataset_path + dataset_str + '_link_feat.npy'))
        return Graph(**init_data)


def get_labels_split_idx(dataset_str: str):
    dataset_path = './datasets/' + dataset_str + '/'
    data_loader = np.load(dataset_path + dataset_str + '_labels.npz')
    labels = torch.LongTensor(data_loader['labels'])
    split_idx = {}
    split_idx['train'] = torch.LongTensor(data_loader['idx_train'])
    split_idx['valid'] = torch.LongTensor(data_loader['idx_val'])
    split_idx['test'] = torch.LongTensor(data_loader['idx_test'])
    return labels, split_idx


def get_split_edge(dataset_str: str):
    dataset_path = './datasets/' + dataset_str + '/'
    split_edge = np.load(dataset_path + dataset_str + '_split_edge.npy', allow_pickle=True)
    return split_edge.item()


def load_dataset(opt, task: str):
    if task == 'link':
        dataset = Graph.load_link_dataset(opt['dataset_str'])
    else:
        dataset = Graph.load_node_dataset(opt['dataset_str'])
    # feature perturbation
    dataset.attr = FeaturePerturbation(
        opt['mechanism'], opt['eps'], opt['m'])(dataset.attr)
    return dataset

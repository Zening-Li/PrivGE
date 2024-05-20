# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class LinkPredictor(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                    hidden_channels: int, num_layers: int, drop_prob: float):
        super(LinkPredictor, self).__init__()

        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.fcs.append(nn.Linear(hidden_channels, hidden_channels))
        self.fcs.append(nn.Linear(hidden_channels, out_channels))
        self.dropout = nn.Dropout(drop_prob)
        self.act_fn = nn.ReLU()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for fc in self.fcs[:-1]:
            x = self.act_fn(fc(x))
            x = self.dropout(x)
        x = self.fcs[-1](x)
        return torch.sigmoid(x)

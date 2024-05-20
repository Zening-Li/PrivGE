# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from layers.dense import Dense


class NodeClassifier(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, 
                    hidden_channels: int, num_layers: int, drop_prob: float):
        super(NodeClassifier, self).__init__()

        self.fcs = nn.ModuleList()
        self.fcs.append(Dense(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.fcs.append(Dense(hidden_channels, hidden_channels))
        self.fcs.append(Dense(hidden_channels, out_channels))

        self.dropout = nn.Dropout(drop_prob)
        self.act_fn = nn.ReLU()

    def forward(self, x: torch.FloatTensor):
        for fc in self.fcs[:-1]:
            x = self.dropout(x)
            x = self.act_fn(fc(x))
        x = self.dropout(x)
        x = self.fcs[-1](x)
        return x

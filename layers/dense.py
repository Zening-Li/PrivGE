# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class Dense(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(Dense, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))

    def forward(self, input: torch.FloatTensor):
        output = torch.mm(input, self.weight)
        if self.in_channels == self.out_channels:
            output = output + input
        return output

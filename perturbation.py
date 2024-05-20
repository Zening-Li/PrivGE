# -*- coding: utf-8 -*-

import numpy as np

import torch
from mechanisms import supported_mechanisms


class FeaturePerturbation:
    def __init__(self, mechanism: str, eps: float, sample_dimensions: int = None):
        self.mechanism = mechanism
        self.eps = eps
        self.sample_dimensions = sample_dimensions

    def __call__(self, x: torch.FloatTensor):
        if np.isinf(self.eps):
            return x
        x = supported_mechanisms[self.mechanism](self.eps, self.sample_dimensions)(x)
        return x

# -*- coding: utf-8 -*-

import math
import torch


class MinMaxScaler:
    def __init__(self, x: torch.FloatTensor):
        # transform the features within a specific range -1., 1.
        self.data_min = x.min(dim=0)[0]
        self.data_max = x.max(dim=0)[0]
        self.delta = self.data_max - self.data_min
        self.delta_zero_index = torch.nonzero(self.delta==0, as_tuple=False).squeeze()
        self.delta_zero_value = x[:, self.delta_zero_index]

    def transform(self, x: torch.FloatTensor):
        trans_feat = (x - self.data_min) * 2 / self.delta - 1.
        trans_feat[:, self.delta_zero_index] = self.delta_zero_value
        return trans_feat

    def recover(self, x: torch.FloatTensor):
        return (x + 1.) * self.delta / 2. + self.data_min


class Mechanism:
    def __init__(self, eps: float, m: int = None):
        self.eps = eps
        self.m = m

    def __call__(self, x: torch.FloatTensor):
        raise NotImplementedError("__call__: not implemented!")


class Laplace(Mechanism):
    def __call__(self, x: torch.FloatTensor):
        scaler = MinMaxScaler(x)
        x = scaler.transform(x)
        dimension = x.size(1)
        # Sensitivity
        s = 2 * dimension
        scale = torch.ones_like(x) * (s / self.eps)
        x_prime = torch.distributions.Laplace(x, scale).sample()
        x_prime = scaler.recover(x_prime)
        return x_prime


class MultiBit(Mechanism):
    def __call__(self, x: torch.FloatTensor):
        scaler = MinMaxScaler(x)
        x = scaler.transform(x)
        dimension = x.size(1)
        m = int(max(1, min(dimension, math.floor(self.eps / 2.18))))

        # sample features for perturbation
        bigS = torch.rand_like(x).topk(m, 1).indices
        s = torch.zeros_like(x, dtype=torch.bool).scatter(1, bigS, True)

        # perturb sampled features
        em = math.exp(self.eps / m)
        p = 1 / (em + 1) * (1 + (x + 1.) / 2. * (em - 1))
        t = torch.bernoulli(p)
        x_star = s * (2 * t - 1)

        # make the result unbiased
        x_prime = dimension * 2. / (2 * m)
        x_prime = x_prime * (em + 1) / (em - 1) * x_star
        x_prime = s * scaler.recover(x_prime)
        
        return x_prime


class Piecewise(Mechanism):
    def __call__(self, x: torch.FloatTensor):
        # piecewise mechanism's variables
        C = (math.exp(self.eps / 2) + 1) / (math.exp(self.eps / 2) - 1)
        p = (math.exp(self.eps) - math.exp(self.eps / 2)) / (2 * math.exp(self.eps / 2) + 2)
        L = (C + 1) / 2 * x - (C - 1) / 2
        R = L + C - 1

        # thresholds for random sampling
        threshold_left = p / math.exp(self.eps) * (L + C)
        threshold_right = threshold_left + p * (R - L)

        # masks for piecewise random sampling
        t = torch.rand_like(x)
        mask_left = t < threshold_left
        mask_middle = (t >= threshold_left) & (t <= threshold_right)
        mask_right = t > threshold_right

        # random sampling
        x = mask_left * (torch.rand_like(x) * (L + C) - C)
        x += mask_middle * (torch.rand_like(x) * (R - L) + L)
        x += mask_right * (torch.rand_like(x) * (C - R) + R)
        return x


class MultiDimPiecewise(Piecewise):
    def __call__(self, x: torch.FloatTensor):
        scaler = MinMaxScaler(x)
        x = scaler.transform(x)
        dimension = x.size(1)
        m = int(max(1, min(dimension, math.floor(self.eps / 2.5))))
        bigS = torch.rand_like(x).topk(m, 1).indices
        s = torch.zeros_like(x, dtype=torch.bool).scatter(1, bigS, True)
        self.eps /= m

        x_star = super().__call__(x)
        x_prime = s * x_star * dimension / m
        x_prime = s * scaler.recover(x_prime)
        return x_prime


class SquareWave(Mechanism):
    def __call__(self, x: torch.FloatTensor):
        # Square Wave mechanism's variables
        e_eps = math.exp(self.eps)
        b = (self.eps * e_eps - e_eps + 1) / (e_eps * (e_eps - 1 - self.eps))
        p = e_eps / (2 * b * e_eps + 2)

        # thresholds for random sampling
        threshold_left = p / e_eps * (x + 1)
        threshold_right = threshold_left + p * 2 * b

        # masks for square ware random sampling
        t = torch.rand_like(x)
        mask_left = t < threshold_left
        mask_middle = (t >= threshold_left) & (t <= threshold_right)
        mask_right = t > threshold_right

        # random sampling
        x = mask_left * (torch.rand_like(x) * (x + 1) - b - 1)
        x += mask_middle * (torch.rand_like(x) * 2 * b + (x - b))
        x += mask_right * (torch.rand_like(x) * (1 - x) + (x + b))

        return x


class HighDimSquareWave(SquareWave):
    def __call__(self, x: torch.FloatTensor):
        scaler = MinMaxScaler(x)
        x = scaler.transform(x)
        # dimension = x.size(1)
        m = self.m
        bigS = torch.rand_like(x).topk(m, 1).indices
        s = torch.zeros_like(x, dtype=torch.bool).scatter(1, bigS, True)
        self.eps /= m

        x_star = super().__call__(x)
        # x_prime = s * x_star * dimension / m
        x_prime = s * scaler.recover(x_star)

        return x_prime


supported_mechanisms = {
    'lp': Laplace,
    'mb': MultiBit,
    'pm': MultiDimPiecewise,
    'hds': HighDimSquareWave
}

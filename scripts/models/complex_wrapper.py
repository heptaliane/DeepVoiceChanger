# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class ComplexWrapper(nn.Module):
    def __init__(self, dataset, reduction='none'):
        self.dataset = dataset

    def forward(self, x):
        real_pred = self.dataset(torch.narrow(x, 1, 0, 1))
        imag_pred = self.dataset(torch.narrow(x, 1, 1, 1))

        if reduction == 'mean':
            return torch.mean((real_pred, imag_pred))

        return torch.cat((real_pred, imag_pred), dim=1)

# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class FramingWrapper(nn.Module):
    def __init__(self, model, frame_size, dim=0, reduction='none'):
        self.model = model
        self.dim = dim
        self.frame_size = frame_size
        self.reduction = reduction

    def forward(self, x):
        length = x.shape[self.dim]
        n_chunk = torch.ceil(length / self.frame_size)
        pred = list()
        for i in range(n_chunk - 1):
            inp = torch.narrow(x, self.dim, i * self.frame_size,
                               self.frame_size)
            pred.append(self.model(inp))

        # Last chunk
        offset = (n_chunk - 1) * self.frame_size
        inp = torch.narrow(x, self.dim, offset, length - offset)
        pad_size = inp.shape
        pad_size[self.dim] = offset + self.frame_size - length
        pad = torch.zeros(pad_size)
        inp = torch.cat((inp, pad), dim=self.dim)
        pred.append(self.model(inp))

        if self.reduction == 'mean':
            return torch.mean(pred)

        return torch.cat(pred, dim=self.dim)

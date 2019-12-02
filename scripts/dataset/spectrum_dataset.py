# -*- coding: utf-8 -*-
import numpy as np
from torch.utils.data import IterableDataset


class SpectrumDataset(IterableDataset):
    def __init__(self, directory, transform=None):
        self._directory = directory
        self.transform = transform
        self._names = directory.names
        self._iter = iter(self._names)

    def __len__(self):
        return len(self._names)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            name = next(self._iter)
        except StopIteration:
            self._iter = iter(self._names)
            name = next(self._iter)

        path = self._directory.name_to_path(name)
        spectrum = np.load(path).get('spectrum')

        if self.transform is not None:
            spectrum = self.transform(spectrum)

        return {'input': spectrum}


# class CycleGanSpectrumDataset

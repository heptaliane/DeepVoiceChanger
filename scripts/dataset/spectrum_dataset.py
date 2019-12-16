# -*- coding: utf-8 -*-
import random

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


class CycleGanSpectrumDataset(IterableDataset):
    def __init__(self, directory_a, directory_b,
                 transform=None, shuffle=False):
        self._dir_a = directory_a
        self._dir_b = directory_b
        self._names_a = directory_a.names
        self._names_b = directory_b.names

        n_names = min(len(self._names_a), len(self._names_b))
        self._names_a = self._names_a[:n_names]
        self._names_b = self._names_b[:n_names]

        self._iter_a = None
        self._iter_b = None
        self._setup_iterator()

    def _setup_iterator(self):
        if self.shuffle:
            random.shuffle(self._names_a)
            random.shuffle(self._names_b)
        self._iter_a = iter(self._names_a)
        self._iter_b = iter(self._names_b)

    def __len__(self):
        return len(self._names_a)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            name_a = next(self._iter_a)
            name_b = next(self._iter_b)
        except StopIteration:
            self._setup_iterator()
            name_a = next(self._iter_a)
            name_b = next(self._iter_b)

        path_a = self._dir_a.name_to_path(name_a)
        path_b = self._dir_b.name_to_path(name_b)
        spectrum_a = np.load(path_a).get('spectrum')
        spectrum_b = np.load(path_b).get('spectrum')

        if self.transform is not None:
            spectrum_a = self.transform(spectrum_a)
            spectrum_b = self.transform(spectrum_b)

        return {'input': {'a': spectrum_a, 'b': spectrum_b}}

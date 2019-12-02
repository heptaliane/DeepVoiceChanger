# -*- coding: utf-8 -*-
import random

import numpy as np


class RandomSliceTransform():
    def __init__(self, length, axis=0):
        self.length = length
        self.axis = axis

    def __call__(self, spectrum):
        total = spectrum.shape[self.axis]
        max_offset = total - self.length
        offset = random.randint(0, max_offset)

        return np.take(spectrum, range(offset, offset + self.length),
                       axis=self.axis)

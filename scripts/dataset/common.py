# -*- coding: utf-8 -*-
import os
import glob


class DatasetDirectory():
    def __init__(self, dirpath, ext=''):
        # Member paramters
        self._path = dirpath
        self.ext = ext

        # Create directory
        os.makedirs(dirpath, exist_ok=True)

    def __str__(self):
        return os.path.join(self._path, '*%s' % self.ext)

    @property
    def names(self):
        names = glob.glob(os.path.join(self._path, '*%s' % self.ext))
        names = [os.path.basename(name) for name in names]
        if len(self.ext) > 0:
            names = [name[:-len(self.ext)] for name in names]
        return names

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, dirpath):
        self._path = dirpath
        os.makedirs(dirpath, exist_ok=True)

    def name_to_path(self, name):
        return os.path.join(self._path, '%s%s' % (name, self.ext))

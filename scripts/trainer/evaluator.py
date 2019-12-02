# -*- coding: utf-8 -*-
import sys
import os

import numpy as np
import torch

sys.path.append('../')
from revert_spectrum import spectrum_to_wave, save_audio_from_numpy

# Logging
from logging import getLogger, NullHandler
logger = getLogger(__name__)
logger.addHandler(NullHandler())


class BestModelSaver():
    def __init__(self, save_dir, name=None):
        name = 'model_best.pth' if name is None else '%s.pth' % name
        self._dst_path = os.path.join(save_dir, name)
        self._loss = float('inf')

    def load_state_dict(self, state):
        self._loss = state['loss']

    def state_dict(self):
        return dict(loss=self._loss)

    def update(self, loss, model):
        if self._loss > loss:
            self._loss = loss
            torch.save(model.state_dict(), self._dst_path)
            logger.info('Save best model (%s).', self._dst_path)


class SnapshotSaver():
    def __init__(self, save_dir):
        self._dst_path = os.path.join(self.save_dir, 'snapshot_latest.pth')

    def update(self, trainer):
        torch.save(trainer.state_dict(), self._dst_path)


class AudioEvaluator():
    def __init__(self, save_dir, sample_rate,
                 window_overwrap_rate, clipping_threshold):
        self.epoch = 0
        self._save_dir = save_dir
        self.sample_rate = sample_rate
        self.window_overwrap_rate = window_overwrap_rate
        self.clipping_threshold = clipping_threshold
        self._offset = 0

    def load_state_dict(self, state):
        self.epoch = state['epoch']

    def state_dict(self):
        return dict(epoch=self.epoch)

    def _save_output_data(self, dst_path, spectrum):
        signal = spectrum_to_wave(spectrum, self.window_overwrap_rate,
                                  self.clipping_threshold)
        save_audio_from_numpy(dst_path, signal, sample_rate)

    def evaluate(self, epoch, _, pred):
        # Update epoch
        if epoch > self.epoch:
            self.epoch = epoch
            self._offset = 0

        batch_size = list(pred.values())[0].shape[0]
        for k, v in pred.items():
            dst_dir = os.path.join(self._save_dir, k)
            os.makedirs(dst_dir, exist_ok=True)
            v = v.cpu().numpy()
            for i in range(batch_size):
                dst_path = os.path.join(dst_dir, '%04d.wav' % self._offset + i)
                self._save_output_data(dst_path, v[i])

        self._offset += batch_size

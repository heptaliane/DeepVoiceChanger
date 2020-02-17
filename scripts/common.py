# -*- coding: utf-8 -*-
""" Common modules """
import json
import wave
import array
import shutil

import numpy as np


class AudioData():
    """ Audio data container """
    def __init__(self, raw, sample_rate, sample_width, n_channel):
        self.raw = raw
        self.sample_rate = sample_rate
        self.sample_width = sample_width
        self.n_channel = n_channel

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, idx):
        return AudioData(self.raw[idx], self.sample_rate, self.sample_width,
                         self.n_channel)

    def to_numpy(self):
        """ Create audio numpy data from bytes """
        if self.sample_width == 2:
            dtype = np.int16
        elif self.sample_width == 4:
            dtype = np.int32
        else:
            raise ValueError('Sample width %d cannot converted to numpy' %
                             self.sample_width)

        return np.frombuffer(self.raw, dtype)


def read_wave(path):
    """ Read .wav file from given path as a AudioData object """
    with wave.open(path, 'rb') as f:
        audio = AudioData(f.readframes(f.getnframes()),
                          f.getframerate(),
                          f.getsampwidth(),
                          f.getnchannels())
    return audio


def write_wave(path, audio):
    """ Write AudioData object to .wav file"""
    with wave.open(path, 'wb') as f:
        f.setnchannels(audio.n_channel)
        f.setsampwidth(audio.sample_width)
        f.setframerate(audio.sample_rate)
        f.writeframesraw(audio.raw)


def write_wave_from_numpy(path, wave, samp_rate):
    # Sample rate = 2
    wave = wave.astype(np.int16)

    # Convert to byte array
    raw = array.array('h', wave).tobytes()

    # Create audio object
    audio = AudioData(raw, samp_rate, 2, 1)

    # Save as a .wav file
    write_wave(path, audio)


def read_json(path):
    """ Read json file """
    with open(path, 'rt') as f:
        data = json.load(f)
    return data


def write_json(path, data):
    """ Write json file """
    with open(path, 'wt') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def check_dependencies(dep_cmds):
    """ Check native module dependencies"""
    err = list()
    for cmd in dep_cmds:
        if shutil.which(cmd) is None:
            err.append(cmd)

    assert len(err) == 0, 'Requirements are not satisfied (%s)' % ','.join(err)

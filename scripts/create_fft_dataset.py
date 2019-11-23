#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import argparse

import numpy as np

from common import read_wave
from config import load_config

# Logging
from logging import getLogger, INFO
import log_initializer
log_initializer.set_root_level(INFO)
log_initializer.set_fmt()
logger = getLogger(__name__)


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description='Audio converter')
    parser.add_argument('--config', '-c', default='config/default.json',
                         help='Path to configuration file.')
    args = parser.parse_args()
    return args


def compute_fourier_features(signal, clipping_threshold):
    # Fourier transform
    signal = signal * np.hamming(len(signal))
    signal = np.fft.fft(signal)

    # Ignore negative frequency
    f = np.fft.fftfreq(len(signal))
    signal = signal[f >= 0]

    def feature_transform(x):
        v = np.abs(x)
        v[v < clipping_threshold] = clipping_threshold
        v = np.log(v) - np.log(clipping_threshold)
        v[x < 0] = -v[x < 0]
        return v

    # Compute Fourier features
    sig_real = feature_transform(signal.real)
    sig_imag = feature_transform(signal.imag)

    return (sig_real, sig_imag)


def create_audio_spectrogram(audio, window_size, window_overwrap_rate,
                             clipping_threshold):
    # Convert audio data to numpy
    signal = audio.to_numpy()

    # Framing settings
    stride = int(window_size * window_overwrap_rate)
    n_frames = len(signal) // stride - int(1 / window_overwrap_rate) + 1

    # Generate spectrum
    spectrum = list()
    for i in range(n_frames):
        src = signal[stride * i: stride * i + window_size]
        spectrum.append(compute_fourier_features(src, clipping_threshold))
    spectrum = np.asarray(spectrum, dtype=np.float32)

    # Normalize
    spectrum = spectrum / np.max(np.abs(spectrum))

    # Order: real/imaginally, frequency, time
    spectrum = spectrum.transpose(1, 2, 0)

    return spectrum


def split_with_slient_sequence(spectrum, split_threshold):
    cnt = 0
    offset = 0
    chunks = list()
    for i in range(spectrum.shape[2]):
        # Silent frame
        if np.all(spectrum[:, :, i] == 0):
            cnt += 1
        # Silent sequence ends
        elif cnt > 0:
            if cnt >= split_threshold:
                chunks.append(spectrum[:, :, offset: i - cnt])
            cnt = 0

    # Append last block
    if cnt < split_threshold:
        chunks.append(spectrum[:, :, offset:])
    else:
        chunks.append(spectrum[:, :, offset: spectrum.shape[2] - cnt])

    return chunks


def main(argv):
    args = parse_arguments(argv)

    # Load configuration
    config = load_config(args.config)

    # Parameters
    src_dir = config['dataset'].get('norm')
    dst_dir = config['dataset'].get('fft')
    window_size = config['fft'].get('window_size')
    window_overwrap_rate = config['fft'].get('window_overwrap_rate')
    clipping_threshold = config['fft'].get('clipping_threshold')
    split_threshold = config['fft'].get('max_allowed_silent_frames')
    min_length = config['fft'].get('min_allowed_spectrum_length')

    # names
    names = src_dir.names

    # Create spectrum dataset
    idx = 0
    for name in names:
        audio = read_wave(src_dir.name_to_path(name))
        spectrum = create_audio_spectrogram(audio, window_size,
                                            window_overwrap_rate,
                                            clipping_threshold)
        spectrums = split_with_slient_sequence(spectrum, split_threshold)

        spectrums = [s for s in spectrums if s.shape[2] >= min_length]

        # For each sequence
        for spectrum in spectrums:
            dst_path = dst_dir.name_to_path('%06d' % idx)
            np.savez_compressed(dst_path, spectrum=spectrum)
            logger.info('Save "%s" (length: %4d)',
                        dst_path, spectrum.shape[2])
            idx += 1


if __name__ == '__main__':
    main(sys.argv[1:])
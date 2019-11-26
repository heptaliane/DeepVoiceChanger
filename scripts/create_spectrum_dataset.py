#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import argparse

import numpy as np

from common import read_wave
from config import load_config
import stft

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


def fourier_transform(signal):
    # Fourier transform
    signal = signal * np.hanning(len(signal))
    # signal = signal * np.hamming(len(signal))
    signal = np.fft.fft(signal)

    # Ignore negative frequency
    signal = signal[:len(signal) // 2 + 1]

    return (signal.real, signal.imag)


def create_audio_spectrogram(audio, window_size, window_overwrap_rate,
                             clipping_threshold):
    # Convert audio data to numpy
    signal = audio.to_numpy()

    # Framing settings
    stride = int(window_size * window_overwrap_rate)

    # Short time Fourier transform
    spectrum = stft.stft(signal, window_size, stride)

    # Compute Fourier feature
    spectrum = stft.to_feature(spectrum, clipping_threshold)

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
                offset = i
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
            logger.info('Save "%s" (length: %4d; from "%s")',
                        dst_path, spectrum.shape[2], name)
            idx += 1


if __name__ == '__main__':
    main(sys.argv[1:])

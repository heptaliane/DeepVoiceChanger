#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import array
import math

import numpy as np

from config import load_config
from common import write_wave, AudioData

# Logging
from logging import getLogger, INFO
import log_initializer
log_initializer.set_root_level(INFO)
log_initializer.set_fmt()
logger = getLogger(__name__)


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description='Revert spectrum to audio')
    parser.add_argument('--config', '-c', default='config/default.json',
                         help='Path to configuration file.')
    parser.add_argument('--input', '-i', required=True,
                         help='Path to spectrum data npz file.')
    parser.add_argument('--output', '-o', default='out.wav',
                         help='Path to output .wav file')
    args = parser.parse_args()
    return args


def spectrum_to_wave(spectrum, window_overwrap_rate):
    # Get Fourier values
    amplitude = np.log(spectrum.shape[1] - 1)
    signals = np.exp(np.abs(spectrum * amplitude)) - 1.0
    negative_mask = spectrum < 0
    signals[negative_mask] = -signals[negative_mask]
    signals = signals[0] + signals[1] * 1.0j
    signals = np.concatenate((signals, signals[1:-1][::-1]), axis=0)

    # Create window
    window_size = signals.shape[0]
    stride = int(window_size * window_overwrap_rate)
    hamming = np.hamming(window_size)
    n_frames = math.ceil(window_size / stride)
    window = np.hamming(window_size)
    for i in range(1, n_frames):
        window[:window_size - stride * i] += hamming[stride * i:] ** 2
        window[stride * i:] += hamming[:window_size - stride * i] ** 2
    window = hamming / window

    # Invert Fourier transform
    signals = [np.fft.ifft(signals[:, i]) for i in range(signals.shape[1])]

    # Overwrap
    length = stride * (len(signals) - 1) + window_size
    wave = np.zeros((length))

    # Construct wave
    for i in range(len(signals)):
        wave[stride * i: stride * i + window_size] += signals[i].imag * window

    # Tune up the volume
    wave = wave / np.max(wave) * 32000

    return wave


def save_audio_from_numpy(dst_path, wave, samp_rate):
    # Sample rate = 2
    wave = wave.astype(np.int16)

    # Convert to byte array
    raw = array.array('h', wave).tobytes()

    # Create audio object
    audio = AudioData(raw, samp_rate, 2, 1)

    # Save as a .wav file
    write_wave(dst_path, audio)


def main(argv):
    args = parse_arguments(argv)

    # Load configuration
    config = load_config(args.config)

    # Parameters
    window_overwrap_rate = config['fft']['window_overwrap_rate']
    samp_rate = config['audio']['sample_rate']

    # Load data
    spectrum = np.load(args.input).get('spectrum')

    # Invert Fourier transform
    wave = spectrum_to_wave(spectrum, window_overwrap_rate)

    # Output file
    save_audio_from_numpy(args.output, wave, samp_rate)


if __name__ == '__main__':
    main(sys.argv[1:])

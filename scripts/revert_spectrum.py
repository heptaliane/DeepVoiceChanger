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
import stft

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


def spectrum_to_wave(spectrum, window_overwrap_rate, clipping_threshold):
    # Compute spectrum from feature
    spectrum = stft.from_feature(spectrum, clipping_threshold)

    # Framing settings
    window_size = spectrum.shape[1] * 2 + 1
    stride = int(window_size * window_overwrap_rate)

    # Short time invert Fourier transform
    signal = stft.stift(spectrum, stride)

    return signal


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
    clipping_threshold = config['fft'].get('clipping_threshold')

    # Load data
    spectrum = np.load(args.input).get('spectrum')

    # Invert Fourier transform
    wave = spectrum_to_wave(spectrum, window_overwrap_rate, clipping_threshold)

    # Output file
    save_audio_from_numpy(args.output, wave, samp_rate)


if __name__ == '__main__':
    main(sys.argv[1:])

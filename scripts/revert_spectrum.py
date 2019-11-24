#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import array

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


def spectrum_to_wave(spectrum, window_overwrap_rate, amplitude):
    # Get Fourier values
    signals = np.exp(np.abs(spectrum * amplitude)) - 1.0
    negative_mask = spectrum < 0
    signals[negative_mask] = -signals[negative_mask]
    signals = signals[0] + signals[1] * 1.0j
    signals = np.concatenate((signals, signals[1:-1][::-1]), axis=0)

    # Invert Fourier transform
    signals = [np.fft.ifft(signals[:, i]) for i in range(signals.shape[1])]

    # Overwrap
    fsize = len(signals[0])
    stride = int(fsize * window_overwrap_rate)
    length = stride * (len(signals) - 1) + fsize
    wave = np.zeros((length))

    # Construct wave
    for i in range(len(signals)):
        wave[stride * i: stride * i + fsize] += signals[i].real

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
    amplitude = 14.6

    # Load data
    spectrum = np.load(args.input).get('spectrum')

    # Invert Fourier transform
    wave = spectrum_to_wave(spectrum, window_overwrap_rate, amplitude)

    # Output file
    save_audio_from_numpy(args.output, wave, samp_rate)


if __name__ == '__main__':
    main(sys.argv[1:])

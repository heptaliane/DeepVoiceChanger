#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description='Spectrum viewer')
    parser.add_argument('--input', '-i', required=True,
                         help='Path to spectrum data npz file.')
    args = parser.parse_args()
    return args


def show_heatmap(spectrum):
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.set_title('Real part')
    ax2.set_title('Imaginally part')

    ax1.imshow(np.abs(spectrum[0]), vmin=0, vmax=1, cmap='jet')
    ax2.imshow(np.abs(spectrum[1]), vmin=0, vmax=1, cmap='jet')

    plt.show()


def main(argv):
    args = parse_arguments(argv)

    spectrum = np.load(args.input).get('spectrum')
    show_heatmap(spectrum)


if __name__ == '__main__':
    main(sys.argv[1:])

# -*- coding: utf-8 -*-
import numpy as np


def stft(signal, window_size, stride):
    # Normalize
    signal = signal.astype(np.float32) / np.max(signal)

    # Fourier transform for each short frame
    n_frames = (len(signal) - window_size) // stride + 1
    window = np.hamming(window_size)
    spectrum = list()
    for i in range(n_frames):
        frame = signal[stride * i: stride * i + window_size] * window
        famp = np.fft.fft(frame)
        spectrum.append((famp.real, famp.imag))

    # To numpy array
    spectrum = np.asarray(spectrum, dtype=np.float32)

    # Remove redundant frequency
    spectrum = spectrum[:, :, :spectrum.shape[2] // 2 + 1]

    # Real/Imaginally, frequency, time
    return spectrum.transpose(1, 2, 0)


def stift(spectrum, stride):
    # To complex value
    spectrum = spectrum[0] + spectrum[1] * 1.0j

    # Add redundant frequency
    spectrum = np.concatenate([spectrum, np.conjugate(spectrum[1:-1][::-1])],
                              axis=0)

    # Invert Fourier transform
    window_size = spectrum.shape[0]
    window = np.hamming(window_size)
    signal = [np.fft.ifft(spectrum[:, i]) * window
              for i in range(spectrum.shape[1])]

    # Overwrap signals
    length = stride * (len(signal) - 1) + window_size
    x = np.zeros(length)
    for i in range(len(signal)):
        x[stride * i: stride * i + window_size] += signal[i].real

    # Tune up the volume
    x = (x.astype(np.float32) / np.max(x)) * 32000

    return x.astype(np.int16)


def to_feature(spectrum, clipping):
    x = np.abs(spectrum)
    x[x < clipping] = clipping
    x = np.log(x) - np.log(clipping)
    neg_mask = spectrum < 0
    x[neg_mask] = -x[neg_mask]
    x = x / np.log(spectrum.shape[1] * 0.5) # Normalize
    return x


def from_feature(feature, clipping=1.0):
    x = feature * np.log(feature.shape[1] * 0.5)
    x = np.exp(np.abs(x)) * clipping
    neg_mask = feature < 0
    x[neg_mask] = -x[neg_mask]
    return x

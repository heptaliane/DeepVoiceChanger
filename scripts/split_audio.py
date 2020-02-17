#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import argparse
import pickle
import os
import tkinter as tk

import numpy as np
import sounddevice as sd

from common import read_wave, write_wave_from_numpy


def get_voice_sequence(audio,
                       amplitude_threshold=1000,
                       split_duration_threshold=0.1,
                       minimum_chunk_duration=0.3):
    arr = audio.to_numpy()

    # Search zero sequence
    indices = np.arange(0, arr.shape[0])
    indices = indices[np.abs(arr) < amplitude_threshold]
    indices -= np.arange(0, indices.shape[0])
    offsets, counts = np.unique(indices, return_counts=True)

    # Compute zero sequence
    duration_th = int(audio.sample_rate * split_duration_threshold)
    chunk_th = int(audio.sample_rate * minimum_chunk_duration)
    cnt = 0
    prev_start_idx, prev_end_idx = 0, 0
    blanks = list()
    for offset, length in zip(offsets, counts):
        start_idx = offset + cnt
        end_idx = start_idx + length
        cnt += length

        if prev_end_idx - prev_start_idx > duration_th:
            if len(blanks) > 0 and prev_start_idx - blanks[-1][1] < chunk_th:
                blanks[-1] = (blanks[-1][1], prev_end_idx)
            else:
                blanks.append((prev_start_idx, prev_end_idx))
        prev_start_idx, prev_end_idx = start_idx, end_idx
    if prev_end_idx > 0:
        blanks.append((prev_start_idx, prev_end_idx))

    # Extract non zero sequence
    indices = [slice(blanks[i - 1][1], blanks[i][0])
               for i in range(1, len(blanks))]
    if blanks[0][0] > 0:
        indices = [slice(0, blanks[0][0]), *indices]
    return indices


class AudioSplitManager():
    def __init__(self, audio, indices, save_dir, swap_name='swap.dat'):
        self._audio = audio
        self._np_audio = audio.to_numpy()
        self._indices = indices
        self._stack = list()
        self._save_dir = save_dir
        self.swap_name = swap_name
        self._curr = 0
        os.makedirs(save_dir, exist_ok=True)
        if os.path.exists(swap_name):
            with open(swap_name, 'rb') as f:
                self.load_state(pickle.load(f))

    def __len__(self):
        return len(self._indices)

    def state(self):
        return {
            'indices': self._indices,
            'stack': self._stack,
            'current': self._curr,
        }

    def load_state(self, state):
        self._indices = state.get('indices')
        self._stack = state.get('stack')
        self._curr = state.get('current')

    @property
    def current_node(self):
        return self._curr

    @property
    def audio(self):
        indices = self._indices[self._curr]
        return self._np_audio[indices]

    @property
    def start_time(self):
        indices = self._indices[self._curr]
        return indices.start / self._audio.sample_rate

    @property
    def end_time(self):
        indices = self._indices[self._curr]
        return indices.stop / self._audio.sample_rate

    @property
    def sample_rate(self):
        return self._audio.sample_rate

    def accept(self):
        dst_path = os.path.join(self._save_dir, '%05d.wav' % self._curr)
        write_wave_from_numpy(dst_path, self.audio, self._audio.sample_rate)
        self._curr += 1

    def reject(self):
        self._stack.append(self._indices.pop(self._curr))

    def undo(self):
        prev = self._stack.pop()
        self._indices.insert(self._curr, prev)

    def concat(self):
        indices = self._indices[self._curr]
        next_indices = self._indices.pop(self._curr + 1)
        self._indices[self._curr] = slice(indices.start, next_indices.stop)

    def __del__(self):
        with open(self.swap_name, 'wb') as f:
            pickle.dump(self.state(), f)


class AudioPlayer():
    def __init__(self, audio_manager):
        self._audio = audio_manager

        self._root = tk.Tk()
        self._root.bind('<Key>', self._key_press)
        self._root.title('Split audio manager')
        self._start_time = tk.StringVar(self._root, '')
        self._end_time = tk.StringVar(self._root, '')
        self._progress = tk.StringVar(self._root, '')
        self._update_indicator()
        self._create_widget()

    def run(self):
        self._root.mainloop()

    def _create_widget(self):
        lbl_container = tk.Frame(self._root)
        lbl_container.pack(expand=1)
        ctl_container = tk.Frame(self._root)
        ctl_container.pack(expand=1)

        opt = dict(side=tk.LEFT, expand=1, fill=tk.X)

        tk.Label(self._root, textvariable=self._progress).pack(expand=1)
        tk.Label(lbl_container, textvariable=self._start_time).pack(**opt)
        tk.Label(lbl_container, text=' - ')
        tk.Label(lbl_container, textvariable=self._end_time).pack(**opt)

        tk.Button(ctl_container, text='Play (Space)',
                  command=self._play_audio).pack(**opt)
        tk.Button(ctl_container, text='Concat (c)',
                  command=self._concat_next_audio).pack(**opt)
        tk.Button(ctl_container, text='Accept (Enter)',
                  command=self._accept_audio).pack(**opt)
        tk.Button(ctl_container, text='Reject (Esc)',
                  command=self._reject_audio).pack(**opt)
        tk.Button(ctl_container, text='Cancel rejection (u)',
                  command=self._cancel_rejection).pack(**opt)

    def _key_press(self, key_event):
        if key_event.keysym == 'space':
            self._play_audio()
        elif key_event.keysym == 'c':
            self._concat_next_audio()
        elif key_event.keysym == 'Return':
            self._accept_audio()
        elif key_event.keysym == 'Escape':
            self._reject_audio()
        elif key_event.keysym == 'u':
            self._cancel_rejection()

    def _update_indicator(self):
        self._start_time.set('%.3f s ' % self._audio.start_time)
        self._end_time.set('%.3f s ' % self._audio.end_time)
        progress = '%d / %d' % (self._audio.current_node, len(self._audio))
        self._progress.set(progress)
        sd.stop()
        self._play_audio()

    def _play_audio(self):
        sd.play(self._audio.audio, self._audio.sample_rate)

    def _concat_next_audio(self):
        self._audio.concat()
        self._update_indicator()

    def _accept_audio(self):
        self._audio.accept()
        self._update_indicator()

    def _reject_audio(self):
        self._audio.reject()
        self._update_indicator()

    def _cancel_rejection(self):
        self._audio.undo()
        self._update_indicator()


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description='Interactive audio splitter')
    parser.add_argument('--input', '-i', required=True,
                        help='Path to split audio')
    parser.add_argument('--output', '-o', required=True,
                        help='Path to output directory')
    parser.add_argument('--device', '-d', default=None,
                        help='Output device id')
    parser.add_argument('--noise_amplitude', '-a', type=int, default=1000,
                        help='Max noise amplitude')
    parser.add_argument('--quiet_duration', '-q', type=float, default=0.3,
                        help='Max quiet duration to allow')
    parser.add_argument('--min_chunk', '-c', type=float, default=0.3,
                        help='Minimum chunk audio length')
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_arguments(argv)

    swap_name = '.%s.swp' % os.path.basename(args.input)

    audio = read_wave(args.input)
    indices = get_voice_sequence(audio, args.noise_amplitude,
                                 args.quiet_duration, args.min_chunk)
    audio_manager = AudioSplitManager(audio, indices, args.output, swap_name)

    if args.device:
        sd.default.device[1] = int(args.device)
    else:
        device = int(input('Device id ? > '))
        sd.default.device[1] = device

    player = AudioPlayer(audio_manager)
    player.run()


if __name__ == '__main__':
    main(sys.argv[1:])

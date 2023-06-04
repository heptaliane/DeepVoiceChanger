#!/usr/local/env python
# -*- coding: utf-8 -*-
import math
import os
from logging import INFO, getLogger
from typing import Dict, List, Optional, Tuple

import click
import log_initializer
import torch
import torchaudio

log_initializer.set_fmt()
log_initializer.set_root_level(INFO)
logger = getLogger(__name__)


def load_audio(
    filename: str,
    bundle: torchaudio.pipelines.Wav2Vec2Bundle,
) -> torch.Tensor:
    wave, sample_rate = torchaudio.load(filename)

    if sample_rate != bundle.sample_rate:
        wave = torchaudio.functional.resample(
            wave,
            sample_rate,
            bundle.sample_rate,
        )

    return wave


class AudioSeparator:
    def __init__(
        self,
        device: torch.device,
        bundle: torchaudio.pipelines.SourceSeparationBundle,
        segment: float = 1.0,
        overlap: float = 0.1,
        batch_size: int = 32,
    ):
        self._device = device
        self._model = bundle.get_model().to(device)
        self._chunk_size = int(bundle.sample_rate * segment * (1 + overlap))
        self._overlap_frames = int(bundle.sample_rate * overlap)
        self._batch_size = batch_size

    def _get_chunk_frames(self, length: int) -> List[Tuple[int, int]]:
        chunks = math.floor(length / self._chunk_size)
        frames = [
            (i * self._chunk_size - self._overlap_frames, (i + 1) * self._chunk_size)
            for i in range(chunks)
        ]
        frames[0] = (0, frames[0][1])
        frames[-1] = (frames[-1][0], length)

        return frames

    def _create_batch(self, audio: torch.Tensor) -> List[torch.Tensor]:
        channels, length = audio.shape
        frames = self._get_chunk_frames(length)

        chunks = [audio[:, start:end] for (start, end) in frames]
        mid_chunks = chunks[1:-1]

        return [
            chunks[0].unsqueeze(dim=0),
            *[
                torch.stack(
                    mid_chunks[i * self._batch_size : (i + 1) * self._batch_size],
                    dim=0,
                )
                for i in range(math.ceil(len(mid_chunks) / self._batch_size))
            ],
            chunks[-1].unsqueeze(dim=0),
        ]

    def _concat_batch(self, batches: List[torch.Tensor], length: int) -> torch.Tensor:
        chunks = list()
        for batch in batches:
            chunks.extend(batch)

        labels, channels, _ = chunks[0].shape
        output = torch.zeros(labels, channels, length, device=self._device)

        fade = torchaudio.transforms.Fade(
            fade_in_len=0,
            fade_out_len=self._overlap_frames,
            fade_shape="linear",
        )

        frames = self._get_chunk_frames(length)
        for i, (start, end) in enumerate(frames[:-1]):
            output[:, :, start:end] += fade(chunks[i])
            if i == 0:
                fade.fade_in_len = self._overlap_frames
        fade.fade_out_len = 0
        start, end = frames[-1]
        output[:, :, start:end] = fade(chunks[-1])

        return output

    def __call__(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        audio = audio.to(self._device)

        # Normalize
        ref = audio.mean(0)
        audio = (audio - ref.mean()) / ref.std()

        batches = self._create_batch(audio)
        outputs = list()
        for batch in batches:
            with torch.no_grad():
                outputs.append(self._model.forward(batch))
        result = self._concat_batch(outputs, audio.shape[-1])

        # Normalize with ref
        result = result * ref.std() + ref.mean()

        return dict(zip(self._model.sources, list(result)))


@click.command()
@click.option("--inputs", "-i", multiple=True, help="Source audio files")
@click.option("--output", "-o", default="data/extracted", help="Output base directory")
@click.option("--extension", "-e", default=None, help="Output file type")
@click.option(
    "--targets",
    "-t",
    multiple=True,
    type=click.Choice(["drums", "bass", "vocals", "other"]),
    default=["vocals"],
    help="Extract targets",
)
def main(inputs: List[str], output: str, extension: Optional[str], targets: List[str]):
    logger.info("Extract targets: [%s]", ", ".join(targets))

    os.makedirs(output, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    bundle = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS

    separator = AudioSeparator(device, bundle)

    for filename in inputs:
        logger.info("Target file: '%s'", filename)

        audio = load_audio(filename, bundle)
        separated = separator(audio)

        result = torch.zeros(audio.shape)
        for target in targets:
            result += separated[target].cpu()

        name, ext = os.path.splitext(os.path.basename(filename))
        if extension is None:
            extension = ext
        dstpath = os.path.join(output, f"{name}{extension}")
        torchaudio.save(dstpath, result, bundle.sample_rate)
        logger.info("Save extracted audio: '%s'", dstpath)


if __name__ == "__main__":
    torch.random.manual_seed(0)
    main()

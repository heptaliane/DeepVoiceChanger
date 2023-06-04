#!/usr/local/env python
# -*- coding: utf-8 -*-
import os
from logging import INFO, getLogger
from typing import List, Dict

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
    ):
        self._device = device
        self._model = bundle.get_model().to(device)
        self._chunk_size = int(bundle.sample_rate * segment * (1 + overlap))
        self._overlap_frames = int(bundle.sample_rate * overlap)

    def __call__(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        audio = audio.to(self._device)

        # Normalize
        ref = audio.mean()
        audio = (audio - ref.mean()) / ref.std()

        batch, channels, length = audio.shape
        result = torch.zeros(
            batch,
            len(self._model.sources),
            channels,
            length,
            device=self._device,
        )
        fade = torchaudio.transforms.Fade(
            fade_in_len=0,
            fade_out_len=self._overlap_frames,
            fade_shape="linear",
        )

        start, end = 0, self._chunk_size
        while start < length - self._overlap_frames:
            # Forward model
            chunk = audio[:, :, start:end]
            with torch.no_grad():
                output = self._model.forward(chunk)
            output = fade(output)
            result[:, :, :, start:end] += output

            # Move to next chunk
            if start == 0:
                fade.fade_in_len = self._overlap_frames
                start += self._chunk_size - self._overlap_frames
            else:
                start += self._chunk_size
            end += self._chunk_size
            if end >= length:
                fade.fade_out_len = 0

        # Normalize with ref
        result = result * ref.std() + ref.mean()

        return dict(zip(self._model.sources, list(result)))


@click.command()
@click.option("--inputs", "-i", multiple=True, help="Source audio files")
@click.option("--output", "-o", default="data/extracted", help="Output base directory")
@click.option(
    "--targets",
    "-t",
    multiple=True,
    type=click.Choice(["drums", "bass", "vocals", "other"]),
    default=["vocals"],
    help="Extract targets",
)
def main(inputs: List[str], output: str, targets: List[str]):
    logger.info("Extract targets: [%s]", ", ".join(targets))

    os.makedirs(output, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    bundle = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS

    separator = AudioSeparator(device, bundle)

    for filename in inputs:
        logger.info("Target file: '%s'", filename)

        audio = load_audio(filename, bundle)
        logger.info("%s", str(audio.shape))
        result = separator(audio)

        output = torch.zeros(audio.shape)
        for target in targets:
            output += result[target].cpu()

        dstpath = os.path.join(output, os.path.basename(filename))
        torchaudio.save(dstpath, output, bundle.sample_rate)
        logger.info("Save extracted audio: '%s'", dstpath)


if __name__ == "__main__":
    torch.random.manual_seed(0)
    main()

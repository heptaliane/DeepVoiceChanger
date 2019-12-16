# -*- coding: utf-8 -*-
import os
from logging import getLogger, NullHandler

import torch

from .voice_converter import VCGenerator, VCDiscriminator
from .complex_wrapper import ComplexWrapper
from .framing_wrapper import FramingWrapper


logger = getLogger(__name__)
logger.addHandler(NullHandler())


def _reshape_state_dict(src, target):
    assert src.dim() == target.dim()
    for d in range(src.dim()):
        chunk = torch.chunk(src, src.shape[d], dim=d)
        while len(chunk) < target.shape[d]:
            chunk.extend(chunk)
        src = torch.cat(chunk[:target.shape[d]], dim=d)
    return src


def load_pretrained_model(model, pretrained_path):
    if not os.path.exists(pretrained_path):
        return

    logger.info('Load pretrained model (%s)', path)
    src = torch.load(pretrained_path)
    dst = model.state_dict()

    state = dict()
    for k in dst.keys():
        if k not in src:
            state[k] = dst[k]
        elif src[k].shape == dst[k].shape:
            state[k] = src[k]
        else:
            state[k] = _reshape_state_dict(src, dst)

    model.load_state_dict(state)


def _extract_used_kwargs(kwargs, labels):
    return {k:kwargs[k] for k in labels if kwargs[k] is not None}


def create_model(architecture, pretrained_path='',
                 use_comp=False, frame_size=-1, **kwargs):
    if architecture.lower() == 'vc_generator':
        logger.info('Create VCGenerator.')
        vc_kwargs = _extract_used_kwargs(('n_channels', 'n_blocks'))
        model = VCGenerator(**vc_kwargs)
        load_pretrained_model(model, pretrained_path)

    elif architecture.lower() == 'vc_discriminator':
        logger.info('Create VCDiscriminator.')
        vc_kwargs = _extract_used_kwargs(('n_blocks'))
        model = VCDiscriminator(**kwargs)
        load_pretrained_model(model, pretrained_path)

    else:
        raise ValueError('Unknown architecture: "%s"' % architecture)

    if use_comp:
        c_kwargs = _extract_used_kwargs(('reduction'))
        model = ComplexWrapper(model, **kwargs)

    if frame_size > 0:
        f_kwargs = _extract_used_kwargs(('dim', 'reduction'))
        model = FramingWrapper(model, frame_size, **f_kwargs)

    return model

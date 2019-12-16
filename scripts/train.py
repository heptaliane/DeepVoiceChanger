#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import argparse

from torch.utils.data import DataLoader
from torch.optim import Adam

from config import load_config
from dataset import CycleGanSpectrumDataset, DatasetDirectory
from model import create_model
from trainer import CycleGanTrainer, AudioEvaluator
from transform import RandomSliceTransform

# Logging
from logging import getLogger, INFO
import log_initializer
log_initializer.set_root_level(INFO)
log_initializer.set_fmt()
logger = getLogger(__name__)


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description='Audio CycleGAN trainer')
    parser.add_argument('--config', '-c', default='config/default.json',
                        help='Path to configuration file.')
    parser.add_argument('--out', '-o', default='result',
                        help='Path to output directory.')
    parser.add_argument('--labels', '-l', required=True, nargs=2,
                        help='Voice label pair to convert.')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU id (Negative value indicates use cpu)')
    args = parser.parse_args()
    return args


def setup_dataset(config, labels):
    transform = RandomSliceTransform(config['fft']['spectrum_frame_size'], 2)

    train_dir = config['dataset']['train']
    train_dir_a = DatasetDirectory(train_dir.name_to_path(lebels[0]))
    train_dir_b = DatasetDirectory(train_dir.name_to_path(lebels[1]))
    train_dataset = CycleGanSpectrumDataset(train_dir_a, train_dir_b,
                                            transform=transform, shuffle=True)
    train_loader = DataLoader(train_dataset, **config['loader'])

    test_dir = config['dataset']['test']
    test_dit_a = DatasetDirectory(test_dir.name_to_path(labels[0]))
    test_dit_b = DatasetDirectory(test_dir.name_to_path(labels[1]))
    test_dataset = CycleGanSpectrumDataset(test_dir_a, test_dir_b,
                                           shuffle=False)
    test_loader = DataLoader(test_dataset, **config['loader'])

    return (train_loader, test_loader)


def setup_model(config, device):
    n_channels = config['fft']['window_size'] // 2 + 1
    gen_kwargs = config['model'].get('generator')
    dis_kwargs = config['model'].get('discriminator')
    common_kwargs = config['model'].get('common')

    gen_a2b = create_model(**gen_kwargs, **common_kwargs).to(device)
    gen_b2a = create_model(**gen_kwargs, **common_kwargs).to(device)
    dis_a = create_model(**dis_kwargs, **common_kwargs).to(device)
    dis_b = create_model(**dis_kwargs, **common_kwargs).to(device)

    gen_params = [*list(gen_a2b.parameters()),
                  *list(gen_b2a.parameters())]
    gen_lr = config['model']['generator'].get('lr')
    gen_optimizer = Adam(gen_params, lr=gen_lr)

    dis_params = [*list(dis_a.parameters()),
                  *list(dis_b.parameters())]
    dis_lr = config['model']['discriminator'].get('lr')
    dis_optimizer = Adam(dis_params, lr=dis_lr)

    return (gen_a2b, gen_b2a, dis_a, dis_b, gen_optimizer, dis_optimizer)


def setup_trainer(config, save_dir, labels, device,
                  loaders, models, optimizers):
    evaluator = AudioEvaluator(save_dir, config['audio']['sample_rate'],
                               config['fft']['window_overwrap_rate'],
                               config['fft']['clipping_threshold'])
    trainer = CycleGanTrainer(*loaders, *models, *optimizers, *labels,
                              device=device, evaluator=evaluator)

    return trainer


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    config = load_config(args.config)
    loaders = setup_dataset(config, args.labels)
    device = None if args.gpu < 0 else args.gpu
    models = setup_model(config, device)
    models, optimizers = models[:4], models[4:]
    trainer = setup_trainer(config, args.out, args.labels, device,
                            loaders, models, optimizers)

    trainer.run()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import subprocess

from common import check_dependencies
from config import load_config

# Logging
from logging import getLogger, INFO
import log_initializer
log_initializer.set_root_level(INFO)
log_initializer.set_fmt()
logger = getLogger(__name__)


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description='Audio converter')
    parser.add_argument('--config', '-c', default='config/default.json',
                         help='Path to configuration file.')
    args = parser.parse_args()
    return args


def normalize_audio(src_path, dst_path, samp_rate, channel):
    # Skip if output file exists
    if os.path.exists(dst_path):
        logger.debug('"%s" exists: skip.', dst_path)
        return

    # Execute native command
    args = ('ffmpeg', '-i', src_path, '-ar', str(samp_rate),
            '-ac', str(channel), '-loglevel', 'error', dst_path)
    stat = subprocess.run(args)

    # Check command result
    if stat.returncode == 0:
        logger.info('"%s" -> "%s"', src_path, dst_path)
    else:
        logger.error('Command terminated with an error. (%s)', src_path)


def main(argv):
    # ffmpeg is required
    check_dependencies(['ffmpeg'])

    # Get Command line arguments
    args = parse_arguments(argv)

    # Load configuration file
    config = load_config(args.config)

    # Import parameters
    src_dirs = config['dataset'].get('org')
    dst_dirs = config['dataset'].get('norm')
    samp_rate = config['audio'].get('sample_rate')
    channel = config['audio'].get('channel')

    for src_dir, dst_dir in zip(src_dirs, dst_dirs):
        # Collect source file name
        names = src_dir.names

        # Normalize audio
        for name in names:
            src_path = src_dir.name_to_path(name)
            dst_path = dst_dir.name_to_path(name)
            normalize_audio(src_path, dst_path, samp_rate, channel)


if __name__ == '__main__':
    main(sys.argv[1:])

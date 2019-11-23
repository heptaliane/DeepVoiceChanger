# -*- coding: utf-8 -*-
from common import read_json
from dataset import DatasetDirectory


def _merge_dict(d1, d2):
    def _merge_list(l1, l2):
        # For each item
        for i in range(len(l2)):
            if len(l1) <= i:
                l1.append(l2[i])
            elif isinstance(l1[i], (dict, list)):
                _merge_dict(l1[i], l2[i])
            else:
                l1[i] = l2[i]

    # For each item
    for k in d2.keys():
        if isinstance(d1[k], dict):
            _merge_dict(d1[k], d2[k])
        elif isinstance(d1[k], list):
            _merge_list(l1[k], l2[k])
        else:
            d1[k] = d2[k]


def _load_dataset_config(dataset_config):
    return {k:DatasetDirectory(v['dirname'], v['ext'])
            for k, v in dataset_config.items()}


def load_config(config_path):
    # Load configuration files
    configs = list()
    while True:
        conf = read_json(config_path)
        configs.append(conf)
        if 'inherit' in conf:
            config_path = conf['inherit']
        else:
            break

    # Merge configurations
    config = configs.pop()
    for ex_conf in configs[::-1]:
        _merge_dict(config, ex_conf)

    # Replace dataset config
    config['dataset'] = _load_dataset_config(config['dataset'])

    return config

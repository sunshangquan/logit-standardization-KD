#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Modified from pycls.
https://github.com/facebookresearch/pycls/blob/main/pycls/datasets/loader.py
"""

import os

import torch
from pycls.core.config import cfg
from pycls.datasets.cifar100 import Cifar100
from pycls.datasets.flowers import Flowers
from pycls.datasets.chaoyang import Chaoyang
from pycls.datasets.tiny_imagenet import TinyImageNet
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler


_DATASETS = {'cifar100': Cifar100, 'flowers': Flowers, "chaoyang": Chaoyang, "tiny_imagenet": TinyImageNet}

_DATA_DIR = os.path.join('.', "data")
if not os.path.exists(_DATA_DIR):
    os.makedirs(_DATA_DIR)

_PATHS = {"cifar100": "", 'flowers': "flowers", "chaoyang": "chaoyang", "tiny_imagenet": "tiny-imagenet-200"}


def _construct_loader(dataset_name, split, batch_size, shuffle, drop_last):
    err_str = "Dataset '{}' not supported".format(dataset_name)
    assert dataset_name in _DATASETS and dataset_name in _PATHS, err_str
    data_path = os.path.join(_DATA_DIR, _PATHS[dataset_name])
    dataset = _DATASETS[dataset_name](data_path, split)
    sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=drop_last,
    )
    return loader


def construct_train_loader():
    return _construct_loader(
        dataset_name=cfg.TRAIN.DATASET,
        split=cfg.TRAIN.SPLIT,
        batch_size=int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=True,
        drop_last=True,
    )


def construct_test_loader():
    return _construct_loader(
        dataset_name=cfg.TEST.DATASET,
        split=cfg.TEST.SPLIT,
        batch_size=int(cfg.TEST.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=False,
        drop_last=False,
    )


def shuffle(loader, cur_epoch):
    err_str = "Sampler type '{}' not supported".format(type(loader.sampler))
    assert isinstance(loader.sampler, (RandomSampler, DistributedSampler)), err_str
    if isinstance(loader.sampler, DistributedSampler):
        loader.sampler.set_epoch(cur_epoch)

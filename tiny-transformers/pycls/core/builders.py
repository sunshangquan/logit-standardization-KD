#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Modified from pycls.
https://github.com/facebookresearch/pycls/blob/main/pycls/core/builders.py
"""

from pycls.core.config import cfg
from pycls.core.net import SoftCrossEntropyLoss
from pycls.models.build import build_model


_loss_funs = {"cross_entropy": SoftCrossEntropyLoss}


def get_loss_fun():
    err_str = "Loss function type '{}' not supported"
    assert cfg.MODEL.LOSS_FUN in _loss_funs.keys(), err_str.format(cfg.TRAIN.LOSS)
    return _loss_funs[cfg.MODEL.LOSS_FUN]


def build_loss_fun():
    return get_loss_fun()()


def register_loss_fun(name, ctor):
    _loss_funs[name] = ctor

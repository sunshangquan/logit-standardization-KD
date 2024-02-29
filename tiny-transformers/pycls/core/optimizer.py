#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Modified from pycls.
https://github.com/facebookresearch/pycls/blob/main/pycls/core/optimizer.py
"""

import torch
from timm.scheduler.cosine_lr import CosineLRScheduler

from pycls.core.config import cfg
from pycls.core.net import unwrap_model


def reset_lr_weight_decay(model):
    body_lr = cfg.OPTIM.BASE_LR
    head_lr = cfg.OPTIM.HEAD_LR_RATIO * body_lr
    skip_list = ['cls_token', 'pos_embed', 'distill_token']
    head_decay = []
    head_no_decay = []
    body_decay = []
    body_no_decay = []

    head_decay_name = []
    head_no_decay_name = []
    body_decay_name = []
    body_no_decay_name = []

    for name, param in unwrap_model(model).named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if name.startswith("head."):
            if len(param.shape) == 1 or name.endswith(".bias"):
                head_no_decay.append(param)
                head_no_decay_name.append(name)
            else:
                head_decay.append(param)
                head_decay_name.append(name)
        else:
            skip = any([k in name for k in skip_list])
            if len(param.shape) == 1 or name.endswith(".bias") or skip:
                body_no_decay.append(param)
                body_no_decay_name.append(name)
            else:
                body_decay.append(param)
                body_decay_name.append(name)
    return [
        {'params': head_no_decay, 'lr': head_lr, 'weight_decay': 0.},
        {'params': head_decay, 'lr': head_lr, 'weight_decay': cfg.OPTIM.WEIGHT_DECAY},
        {'params': body_no_decay, 'lr': body_lr, 'weight_decay': 0.},
        {'params': body_decay, 'lr': body_lr, 'weight_decay': cfg.OPTIM.WEIGHT_DECAY}]


def construct_optimizer(model):
    optim = cfg.OPTIM
    param_wds = reset_lr_weight_decay(model)
    if optim.OPTIMIZER == "sgd":
        optimizer = torch.optim.SGD(
            param_wds,
            lr=optim.BASE_LR,
            momentum=optim.MOMENTUM,
            weight_decay=optim.WEIGHT_DECAY,
            dampening=optim.DAMPENING,
            nesterov=optim.NESTEROV,
        )
    elif optim.OPTIMIZER == "adam":
        optimizer = torch.optim.Adam(
            param_wds,
            lr=optim.BASE_LR,
            betas=(optim.BETA1, optim.BETA2),
            weight_decay=optim.WEIGHT_DECAY,
        )
    elif optim.OPTIMIZER == "adamw":
        optimizer = torch.optim.AdamW(
            param_wds,
            lr=optim.BASE_LR,
            betas=(optim.BETA1, optim.BETA2),
            weight_decay=optim.WEIGHT_DECAY,
        )
    else:
        raise NotImplementedError
    return optimizer


def construct_scheduler(optimizer):
    warmup_lr = cfg.OPTIM.WARMUP_FACTOR * cfg.OPTIM.BASE_LR
    if cfg.OPTIM.LR_POLICY == 'cos':
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=cfg.OPTIM.MAX_EPOCH,
            lr_min=cfg.OPTIM.MIN_LR,
            warmup_t=cfg.OPTIM.WARMUP_EPOCHS,
            warmup_lr_init=warmup_lr)
    else:
        raise NotImplementedError
    return scheduler


def get_current_lr(optimizer):
    return optimizer.param_groups[0]['lr']

#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Modified from pycls.
https://github.com/facebookresearch/pycls/blob/main/pycls/core/net.py
"""

import itertools

import numpy as np
import pycls.core.distributed as dist
import torch
from pycls.core.config import cfg


def unwrap_model(model):
    wrapped = isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel)
    return model.module if wrapped else model


def smooth_one_hot_labels(labels):
    n_classes, label_smooth = cfg.MODEL.NUM_CLASSES, cfg.TRAIN.LABEL_SMOOTHING
    err_str = "Invalid input to one_hot_vector()"
    assert labels.ndim == 1 and labels.max() < n_classes, err_str
    shape = (labels.shape[0], n_classes)
    neg_val = label_smooth / n_classes
    pos_val = 1.0 - label_smooth + neg_val
    labels_one_hot = torch.full(shape, neg_val, dtype=torch.float, device=labels.device)
    labels_one_hot.scatter_(1, labels.long().view(-1, 1), pos_val)
    return labels_one_hot


class SoftCrossEntropyLoss(torch.nn.Module):

    def __init__(self):
        super(SoftCrossEntropyLoss, self).__init__()

    def _cross_entropy(self, x, y):
        loss = -y * torch.nn.functional.log_softmax(x, -1)
        return torch.sum(loss) / x.shape[0]

    def forward(self, x, y):
        if isinstance(x, list):
            losses = [self._cross_entropy(pred, y) / len(x) for pred in x]
            return sum(losses)
        return self._cross_entropy(x, y)


def mixup(inputs, labels):
    assert labels.shape[1] == cfg.MODEL.NUM_CLASSES, "mixup labels must be one-hot"
    mixup_alpha, cutmix_alpha = cfg.TRAIN.MIXUP_ALPHA, cfg.TRAIN.CUTMIX_ALPHA
    mixup_alpha = mixup_alpha if (cutmix_alpha == 0 or np.random.rand() < 0.5) else 0
    if mixup_alpha > 0:
        m = np.random.beta(mixup_alpha, mixup_alpha)
        permutation = torch.randperm(labels.shape[0])
        inputs = m * inputs + (1.0 - m) * inputs[permutation, :]
        labels = m * labels + (1.0 - m) * labels[permutation, :]
    elif cutmix_alpha > 0:
        m = np.random.beta(cutmix_alpha, cutmix_alpha)
        permutation = torch.randperm(labels.shape[0])
        h, w = inputs.shape[2], inputs.shape[3]
        w_b, h_b = np.int(w * np.sqrt(1.0 - m)), np.int(h * np.sqrt(1.0 - m))
        x_c, y_c = np.random.randint(w), np.random.randint(h)
        x_0, y_0 = np.clip(x_c - w_b // 2, 0, w), np.clip(y_c - h_b // 2, 0, h)
        x_1, y_1 = np.clip(x_c + w_b // 2, 0, w), np.clip(y_c + h_b // 2, 0, h)
        m = 1.0 - ((x_1 - x_0) * (y_1 - y_0) / (h * w))
        inputs[:, :, y_0:y_1, x_0:x_1] = inputs[permutation, :, y_0:y_1, x_0:x_1]
        labels = m * labels + (1.0 - m) * labels[permutation, :]
    return inputs, labels, labels.argmax(1)


def update_model_ema(model, model_ema, cur_epoch, cur_iter):
    update_period = cfg.OPTIM.EMA_UPDATE_PERIOD
    if update_period == 0 or cur_iter % update_period != 0:
        return
    adjust = cfg.TRAIN.BATCH_SIZE / cfg.OPTIM.MAX_EPOCH * update_period
    alpha = min(1.0, cfg.OPTIM.EMA_ALPHA * adjust)
    alpha = 1.0 if cur_epoch < cfg.OPTIM.WARMUP_EPOCHS else alpha
    params = unwrap_model(model).state_dict()
    for name, param in unwrap_model(model_ema).state_dict().items():
        param.copy_(param * (1.0 - alpha) + params[name] * alpha)

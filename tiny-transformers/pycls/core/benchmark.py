#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Modified from pycls.
https://github.com/facebookresearch/pycls/blob/main/pycls/core/benchmark.py
"""

import numpy as np
import pycls.core.logging as logging
import pycls.core.net as net
import pycls.datasets.loader as loader
import torch
import torch.cuda.amp as amp
from pycls.core.config import cfg
from pycls.core.timer import Timer


logger = logging.get_logger(__name__)


@torch.no_grad()
def compute_time_eval(model):
    model.eval()
    im_size, batch_size = cfg.MODEL.IMG_SIZE, int(cfg.TEST.BATCH_SIZE / cfg.NUM_GPUS)
    inputs = torch.zeros(batch_size, 3, im_size, im_size).cuda(non_blocking=False)
    timer = Timer()
    total_iter = cfg.PREC_TIME.NUM_ITER + cfg.PREC_TIME.WARMUP_ITER
    for cur_iter in range(total_iter):
        if cur_iter == cfg.PREC_TIME.WARMUP_ITER:
            timer.reset()
        timer.tic()
        model(inputs)
        torch.cuda.synchronize()
        timer.toc()
    return timer.average_time


def compute_time_train(model, loss_fun):
    model.train()
    im_size, batch_size = cfg.MODEL.IMG_SIZE, int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS)
    inputs = torch.rand(batch_size, 3, im_size, im_size).cuda(non_blocking=False)
    labels = torch.zeros(batch_size, dtype=torch.int64).cuda(non_blocking=False)
    labels_one_hot = net.smooth_one_hot_labels(labels)
    offline_features = None
    if hasattr(net.unwrap_model(model), 'guidance_loss') and cfg.DISTILLATION.OFFLINE:
        kd_data = np.load(cfg.DISTILLATION.FEATURE_FILE)
        offline_features = []
        for i in range(len(kd_data.files)):
            feat = torch.from_numpy(kd_data[f'layer_{i}'][0]).cuda(non_blocking=False)
            offline_features.append(feat.unsqueeze(0).repeat(batch_size, 1, 1, 1))
    bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    bn_stats = [[bn.running_mean.clone(), bn.running_var.clone()] for bn in bns]
    scaler = amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)
    fw_timer, bw_timer = Timer(), Timer()
    total_iter = cfg.PREC_TIME.NUM_ITER + cfg.PREC_TIME.WARMUP_ITER
    for cur_iter in range(total_iter):
        if cur_iter == cfg.PREC_TIME.WARMUP_ITER:
            fw_timer.reset()
            bw_timer.reset()
        fw_timer.tic()
        with amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):
            preds = model(inputs)
            loss_cls = loss_fun(preds, labels_one_hot)
            loss, loss_inter, loss_logit = loss_cls, inputs.new_tensor(0.0), inputs.new_tensor(0.0)
            if hasattr(net.unwrap_model(model), 'guidance_loss'):
                loss_inter, loss_logit = net.unwrap_model(model).guidance_loss(inputs, offline_features)
                if cfg.DISTILLATION.ENABLE_LOGIT:
                    loss_cls = loss_cls * (1 - cfg.DISTILLATION.LOGIT_WEIGHT)
                    loss_logit = loss_logit * cfg.DISTILLATION.LOGIT_WEIGHT
                    loss = loss_cls + loss_logit
                if cfg.DISTILLATION.ENABLE_INTER:
                    loss_inter = loss_inter * cfg.DISTILLATION.INTER_WEIGHT
                    loss = loss_cls + loss_inter
        torch.cuda.synchronize()
        fw_timer.toc()
        bw_timer.tic()
        scaler.scale(loss).backward()
        torch.cuda.synchronize()
        bw_timer.toc()
    for bn, (mean, var) in zip(bns, bn_stats):
        bn.running_mean, bn.running_var = mean, var
    return fw_timer.average_time, bw_timer.average_time


def compute_time_loader(data_loader):
    timer = Timer()
    loader.shuffle(data_loader, 0)
    data_loader_iterator = iter(data_loader)
    total_iter = cfg.PREC_TIME.NUM_ITER + cfg.PREC_TIME.WARMUP_ITER
    total_iter = min(total_iter, len(data_loader))
    for cur_iter in range(total_iter):
        if cur_iter == cfg.PREC_TIME.WARMUP_ITER:
            timer.reset()
        timer.tic()
        next(data_loader_iterator)
        timer.toc()
    return timer.average_time


def compute_time_model(model, loss_fun):
    logger.info("Computing model timings only...")
    test_fw_time = compute_time_eval(model)
    train_fw_time, train_bw_time = compute_time_train(model, loss_fun)
    train_fw_bw_time = train_fw_time + train_bw_time
    iter_times = {
        "test_fw_time": test_fw_time,
        "train_fw_time": train_fw_time,
        "train_bw_time": train_bw_time,
        "train_fw_bw_time": train_fw_bw_time,
    }
    logger.info(logging.dump_log_data(iter_times, "iter_times"))


def compute_time_full(model, loss_fun, train_loader, test_loader):
    logger.info("Computing model and loader timings...")
    test_fw_time = compute_time_eval(model)
    train_fw_time, train_bw_time = compute_time_train(model, loss_fun)
    train_fw_bw_time = train_fw_time + train_bw_time
    train_loader_time = compute_time_loader(train_loader)
    iter_times = {
        "test_fw_time": test_fw_time,
        "train_fw_time": train_fw_time,
        "train_bw_time": train_bw_time,
        "train_fw_bw_time": train_fw_bw_time,
        "train_loader_time": train_loader_time,
    }
    logger.info(logging.dump_log_data(iter_times, "iter_times"))
    epoch_times = {
        "test_fw_time": test_fw_time * len(test_loader),
        "train_fw_time": train_fw_time * len(train_loader),
        "train_bw_time": train_bw_time * len(train_loader),
        "train_fw_bw_time": train_fw_bw_time * len(train_loader),
        "train_loader_time": train_loader_time * len(train_loader),
    }
    logger.info(logging.dump_log_data(epoch_times, "epoch_times"))
    overhead = max(0, train_loader_time - train_fw_bw_time) / train_fw_bw_time
    logger.info("Overhead of data loader is {:.2f}%".format(overhead * 100))

#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Modified from pycls.
https://github.com/facebookresearch/pycls/blob/main/pycls/core/meters.py
"""

from collections import deque

import os
import numpy as np
import pycls.core.logging as logging
import torch
from pycls.core.config import cfg
from pycls.core.timer import Timer


def time_string(seconds):
    days, rem = divmod(int(seconds), 24 * 3600)
    hrs, rem = divmod(rem, 3600)
    mins, secs = divmod(rem, 60)
    return "{0:02},{1:02}:{2:02}:{3:02}".format(days, hrs, mins, secs)


def topk_errors(preds, labels, ks):
    if isinstance(preds, list):
        preds = preds[0] + preds[1]
    return _topk_errors(preds, labels, ks)


def _topk_errors(preds, labels, ks):
    err_str = "Batch dim of predictions and labels must match"
    assert preds.size(0) == labels.size(0), err_str
    ks = [min(k, preds.size(-1)) for k in ks]
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )
    top_max_k_inds = top_max_k_inds.t()
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    topks_correct = [top_max_k_correct[:k, :].reshape(-1).float().sum() for k in ks]
    return [(1.0 - x / preds.size(0)) * 100.0 for x in topks_correct]


def gpu_mem_usage():
    mem_usage_bytes = torch.cuda.max_memory_allocated()
    return mem_usage_bytes / 1024 / 1024


class ScalarMeter(object):

    def __init__(self, window_size):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def reset(self):
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def add_value(self, value):
        self.deque.append(value)
        self.count += 1
        self.total += value

    def get_win_median(self):
        return np.median(self.deque)

    def get_win_avg(self):
        return np.mean(self.deque)

    def get_global_avg(self):
        return self.total / self.count


class TrainMeter(object):

    def __init__(self, epoch_iters, phase="train"):
        self.logger = logging.get_logger(__name__)
        log_file = os.path.join(cfg.OUT_DIR, 'log.txt')
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)

        self.tf_logger = logging.get_tflogger()
        self.tf_logger.initialize(cfg.OUT_DIR)

        self.epoch_iters = epoch_iters
        self.max_iter = cfg.OPTIM.MAX_EPOCH * epoch_iters
        self.phase = phase
        self.iter_timer = Timer()
        self.cls_loss = ScalarMeter(cfg.LOG_PERIOD)
        self.inter_loss = ScalarMeter(cfg.LOG_PERIOD)
        self.logit_loss = ScalarMeter(cfg.LOG_PERIOD)
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_total = 0.0
        self.lr = None
        self.mb_top1_err = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_err = ScalarMeter(cfg.LOG_PERIOD)
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0

    def reset(self, timer=False):
        if timer:
            self.iter_timer.reset()
        self.cls_loss.reset()
        self.inter_loss.reset()
        self.logit_loss.reset()
        self.loss.reset()
        self.loss_total = 0.0
        self.lr = None
        self.mb_top1_err.reset()
        self.mb_top5_err.reset()
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0

    def iter_tic(self):
        self.iter_timer.tic()

    def iter_toc(self):
        self.iter_timer.toc()

    def update_stats(self, top1_err, top5_err, cls_loss, inter_loss, logit_loss, loss, lr, mb_size):
        self.mb_top1_err.add_value(top1_err)
        self.mb_top5_err.add_value(top5_err)
        self.cls_loss.add_value(cls_loss)
        self.inter_loss.add_value(inter_loss)
        self.logit_loss.add_value(logit_loss)
        self.loss.add_value(loss)
        self.lr = lr
        self.num_top1_mis += top1_err * mb_size
        self.num_top5_mis += top5_err * mb_size
        self.loss_total += loss * mb_size
        self.num_samples += mb_size

    def get_iter_stats(self, cur_epoch, cur_iter):
        cur_iter_total = cur_epoch * self.epoch_iters + cur_iter + 1
        eta_sec = self.iter_timer.average_time * (self.max_iter - cur_iter_total)
        mem_usage = gpu_mem_usage()
        stats = {
            "epoch": "{}/{}".format(cur_epoch + 1, cfg.OPTIM.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "total_iter": cur_iter_total,
            "time_avg": self.iter_timer.average_time,
            "time_diff": self.iter_timer.diff,
            "eta": time_string(eta_sec),
            "top1_err": self.mb_top1_err.get_win_median(),
            "top5_err": self.mb_top5_err.get_win_median(),
            "cls_loss": self.cls_loss.get_win_median(),
            "loss": self.loss.get_win_median(),
            "lr": self.lr,
            "mem": int(np.ceil(mem_usage)),
        }
        if cfg.DISTILLATION.ENABLE_INTER:
            stats["inter_distill_loss"] = self.inter_loss.get_win_median()
        if cfg.DISTILLATION.ENABLE_LOGIT:
            stats["logit_distill_loss"] = self.logit_loss.get_win_median()
        return stats

    def log_iter_stats(self, cur_epoch, cur_iter):
        if (cur_iter + 1) % cfg.LOG_PERIOD == 0:
            stats = self.get_iter_stats(cur_epoch, cur_iter)
            self.logger.info(logging.dump_log_data(stats, self.phase + "_iter"))
            self.tf_logger.log_iter_stats(stats)

    def get_epoch_stats(self, cur_epoch):
        cur_iter_total = (cur_epoch + 1) * self.epoch_iters
        eta_sec = self.iter_timer.average_time * (self.max_iter - cur_iter_total)
        mem_usage = gpu_mem_usage()
        top1_err = self.num_top1_mis / self.num_samples
        top5_err = self.num_top5_mis / self.num_samples
        avg_loss = self.loss_total / self.num_samples
        stats = {
            "epoch": "{}/{}".format(cur_epoch + 1, cfg.OPTIM.MAX_EPOCH),
            "time_avg": self.iter_timer.average_time,
            "time_epoch": self.iter_timer.average_time * self.epoch_iters,
            "eta": time_string(eta_sec),
            "top1_err": top1_err,
            "top5_err": top5_err,
            "loss": avg_loss,
            "lr": self.lr,
            "mem": int(np.ceil(mem_usage)),
        }
        return stats

    def log_epoch_stats(self, cur_epoch):
        stats = self.get_epoch_stats(cur_epoch)
        self.logger.info(logging.dump_log_data(stats, self.phase + "_epoch"))
        self.tf_logger.log_epoch_stats(stats)


class TestMeter(object):

    def __init__(self, epoch_iters, phase="test"):
        self.logger = logging.get_logger(__name__)
        self.tf_logger = logging.get_tflogger()
        self.epoch_iters = epoch_iters
        self.phase = phase
        self.iter_timer = Timer()
        self.mb_top1_err = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_err = ScalarMeter(cfg.LOG_PERIOD)
        self.min_top1_err = 100.0
        self.min_top5_err = 100.0
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0

    def reset(self, min_errs=False):
        if min_errs:
            self.min_top1_err = 100.0
            self.min_top5_err = 100.0
        self.iter_timer.reset()
        self.mb_top1_err.reset()
        self.mb_top5_err.reset()
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0

    def iter_tic(self):
        self.iter_timer.tic()

    def iter_toc(self):
        self.iter_timer.toc()

    def update_stats(self, top1_err, top5_err, mb_size):
        self.mb_top1_err.add_value(top1_err)
        self.mb_top5_err.add_value(top5_err)
        self.num_top1_mis += top1_err * mb_size
        self.num_top5_mis += top5_err * mb_size
        self.num_samples += mb_size

    def get_iter_stats(self, cur_epoch, cur_iter):
        mem_usage = gpu_mem_usage()
        iter_stats = {
            "epoch": "{}/{}".format(cur_epoch + 1, cfg.OPTIM.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "time_avg": self.iter_timer.average_time,
            "time_diff": self.iter_timer.diff,
            "top1_err": self.mb_top1_err.get_win_median(),
            "top5_err": self.mb_top5_err.get_win_median(),
            "mem": int(np.ceil(mem_usage)),
        }
        return iter_stats

    def log_iter_stats(self, cur_epoch, cur_iter):
        if (cur_iter + 1) % cfg.LOG_PERIOD == 0:
            stats = self.get_iter_stats(cur_epoch, cur_iter)
            self.logger.info(logging.dump_log_data(stats, self.phase + "_iter"))

    def get_epoch_stats(self, cur_epoch):
        top1_err = self.num_top1_mis / self.num_samples
        top5_err = self.num_top5_mis / self.num_samples
        self.min_top1_err = min(self.min_top1_err, top1_err)
        self.min_top5_err = min(self.min_top5_err, top5_err)
        mem_usage = gpu_mem_usage()
        stats = {
            "epoch": "{}/{}".format(cur_epoch + 1, cfg.OPTIM.MAX_EPOCH),
            "time_avg": self.iter_timer.average_time,
            "time_epoch": self.iter_timer.average_time * self.epoch_iters,
            "top1_err": top1_err,
            "top5_err": top5_err,
            "min_top1_err": self.min_top1_err,
            "min_top5_err": self.min_top5_err,
            "mem": int(np.ceil(mem_usage)),
        }
        return stats

    def log_epoch_stats(self, cur_epoch):
        stats = self.get_epoch_stats(cur_epoch)
        self.logger.info(logging.dump_log_data(stats, self.phase + "_epoch"))
        if self.phase == 'test':
            self.tf_logger.log_test_stats(stats)

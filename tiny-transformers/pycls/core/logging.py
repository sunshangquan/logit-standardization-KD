#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Modified from pycls.
https://github.com/facebookresearch/pycls/blob/main/pycls/core/logging.py
"""

from torch.utils.tensorboard import SummaryWriter
import builtins
import decimal
import logging
import os
import sys
from logging import FileHandler, INFO

import pycls.core.distributed as dist
import simplejson
from pycls.core.config import cfg
from pycls.core.io import pathmgr


_FORMAT = "[%(filename)s: %(lineno)3d]: %(message)s"

_LOG_FILE = "stdout.log"

_TAG = "json_stats: "

_TYPE = "_type"


def _suppress_print():
    def ignore(*_objects, _sep=" ", _end="\n", _file=sys.stdout, _flush=False):
        pass

    builtins.print = ignore


def setup_logging():
    if dist.is_main_proc():
        logging.root.handlers = []
        logging_config = {"level": logging.INFO, "format": _FORMAT}
        if cfg.LOG_DEST == "stdout":
            logging_config["stream"] = sys.stdout
        else:
            logging_config["filename"] = os.path.join(cfg.OUT_DIR, _LOG_FILE)
        logging.basicConfig(**logging_config)
    else:
        _suppress_print()


def get_logger(name):
    return logging.getLogger(name)


def dump_log_data(data, data_type, prec=4):
    data[_TYPE] = data_type
    data = float_to_decimal(data, prec)
    data_json = simplejson.dumps(data, sort_keys=True, use_decimal=True)
    return "{:s}{:s}".format(_TAG, data_json)


def float_to_decimal(data, prec=4):
    if prec and isinstance(data, dict):
        return {k: float_to_decimal(v, prec) for k, v in data.items()}
    if prec and isinstance(data, float):
        return decimal.Decimal(("{:." + str(prec) + "f}").format(data))
    else:
        return data


def get_log_files(log_dir, name_filter="", log_file=_LOG_FILE):
    names = [n for n in sorted(pathmgr.ls(log_dir)) if name_filter in n]
    files = [os.path.join(log_dir, n, log_file) for n in names]
    f_n_ps = [(f, n) for (f, n) in zip(files, names) if pathmgr.exists(f)]
    files, names = zip(*f_n_ps) if f_n_ps else ([], [])
    return files, names


def load_log_data(log_file, data_types_to_skip=()):
    assert pathmgr.exists(log_file), "Log file not found: {}".format(log_file)
    with pathmgr.open(log_file, "r") as f:
        lines = f.readlines()
    lines = [l[l.find(_TAG) + len(_TAG) :] for l in lines if _TAG in l]
    lines = [simplejson.loads(l) for l in lines]
    lines = [l for l in lines if _TYPE in l and not l[_TYPE] in data_types_to_skip]
    data_types = [l[_TYPE] for l in lines]
    data = {t: [] for t in data_types}
    for t, line in zip(data_types, lines):
        del line[_TYPE]
        data[t].append(line)
    for t in data:
        metrics = sorted(data[t][0].keys())
        err_str = "Inconsistent metrics in log for _type={}: {}".format(t, metrics)
        assert all(sorted(d.keys()) == metrics for d in data[t]), err_str
        data[t] = {m: [d[m] for d in data[t]] for m in metrics}
    return data


def sort_log_data(data):
    for t in data:
        if "epoch" in data[t]:
            assert "epoch_ind" not in data[t] and "epoch_max" not in data[t]
            data[t]["epoch_ind"] = [int(e.split("/")[0]) for e in data[t]["epoch"]]
            data[t]["epoch_max"] = [int(e.split("/")[1]) for e in data[t]["epoch"]]
            epoch = data[t]["epoch_ind"]
            if "iter" in data[t]:
                assert "iter_ind" not in data[t] and "iter_max" not in data[t]
                data[t]["iter_ind"] = [int(i.split("/")[0]) for i in data[t]["iter"]]
                data[t]["iter_max"] = [int(i.split("/")[1]) for i in data[t]["iter"]]
                itr = zip(epoch, data[t]["iter_ind"], data[t]["iter_max"])
                epoch = [e + (i_ind - 1) / i_max for e, i_ind, i_max in itr]
            for m in data[t]:
                data[t][m] = [v for _, v in sorted(zip(epoch, data[t][m]))]
        else:
            data[t] = {m: d[0] for m, d in data[t].items()}
    return data


class TFLogger(object):

    def __init__(self):
        self.writer = None

    def initialize(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_iter_stats(self, stats):
        if self.writer is None:
            return
        total_iter = stats['total_iter']
        for k in stats:
            if 'loss' in k:
                self.writer.add_scalar(f'iteration/{k}', stats[k], total_iter)
        self.writer.add_scalar('iteration/top1_acc', 100 - stats['top1_err'], total_iter)
        self.writer.add_scalar('iteration/top5_acc', 100 - stats['top5_err'], total_iter)

    def log_epoch_stats(self, stats):
        if self.writer is None:
            return
        epoch = int(stats['epoch'].split('/')[0])
        self.writer.add_scalar('epoch/loss', stats['loss'], epoch)
        self.writer.add_scalar('epoch/learning_rate', stats['lr'], epoch)
        self.writer.add_scalar('epoch/top1_acc', 100 - stats['top1_err'], epoch)
        self.writer.add_scalar('epoch/top5_acc', 100 - stats['top5_err'], epoch)

    def log_test_stats(self, stats):
        if self.writer is None:
            return
        epoch = int(stats['epoch'].split('/')[0])
        self.writer.add_scalar('test/top1_acc', 100 - stats['top1_err'], epoch)
        self.writer.add_scalar('test/top5_acc', 100 - stats['top5_err'], epoch)
        self.writer.add_scalar('test/max_top1_acc', 100 - stats['min_top1_err'], epoch)
        self.writer.add_scalar('test/max_top5_acc', 100 - stats['min_top5_err'], epoch)


_tf_logger = TFLogger()


def get_tflogger():
    return _tf_logger

#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Modified from pycls.
https://github.com/facebookresearch/pycls/blob/main/pycls/core/checkpoint.py
"""

import os

import pycls.core.distributed as dist
import torch
from pycls.core.config import cfg
from pycls.core.io import pathmgr
from pycls.core.net import unwrap_model
import pycls.core.logging as logging


logger = logging.get_logger(__name__)


_NAME_PREFIX = "model_epoch_"

_DIR_NAME = "checkpoints"


def get_checkpoint_dir():
    return os.path.join(cfg.OUT_DIR, _DIR_NAME)


def get_checkpoint(epoch):
    name = "{}{:04d}.pyth".format(_NAME_PREFIX, epoch)
    return os.path.join(get_checkpoint_dir(), name)


def get_checkpoint_best():
    return os.path.join(cfg.OUT_DIR, "model.pyth")


def get_last_checkpoint():
    checkpoint_dir = get_checkpoint_dir()
    checkpoints = [f for f in pathmgr.ls(checkpoint_dir) if _NAME_PREFIX in f]
    last_checkpoint_name = sorted(checkpoints)[-1]
    return os.path.join(checkpoint_dir, last_checkpoint_name)


def has_checkpoint():
    checkpoint_dir = get_checkpoint_dir()
    if not pathmgr.exists(checkpoint_dir):
        return False
    return any(_NAME_PREFIX in f for f in pathmgr.ls(checkpoint_dir))


def save_checkpoint(model, model_ema, optimizer, epoch, test_err, ema_err):
    if not dist.is_main_proc():
        return
    pathmgr.mkdirs(get_checkpoint_dir())
    checkpoint = {
        "epoch": epoch,
        "test_err": test_err,
        "ema_err": ema_err,
        "model_state": unwrap_model(model).state_dict(),
        "ema_state": unwrap_model(model_ema).state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "cfg": cfg.dump(),
    }
    checkpoint_file = get_checkpoint(epoch + 1)
    with pathmgr.open(checkpoint_file, "wb") as f:
        torch.save(checkpoint, f)
    if not pathmgr.exists(get_checkpoint_best()):
        pathmgr.copy(checkpoint_file, get_checkpoint_best())
    else:
        with pathmgr.open(get_checkpoint_best(), "rb") as f:
            best = torch.load(f, map_location="cpu")
        if test_err < best["test_err"] or ema_err < best["ema_err"]:
            if test_err < best["test_err"]:
                best["model_state"] = checkpoint["model_state"]
                best["test_err"] = test_err
            if ema_err < best["ema_err"]:
                best["ema_state"] = checkpoint["ema_state"]
                best["ema_err"] = ema_err
            with pathmgr.open(get_checkpoint_best(), "wb") as f:
                torch.save(best, f)
    return checkpoint_file


def load_checkpoint(checkpoint_file, model, model_ema=None, optimizer=None):
    err_str = "Checkpoint '{}' not found"
    assert pathmgr.exists(checkpoint_file), err_str.format(checkpoint_file)
    with pathmgr.open(checkpoint_file, "rb") as f:
        checkpoint = torch.load(f, map_location="cpu")
    test_err = checkpoint["test_err"] if "test_err" in checkpoint else 100
    ema_err = checkpoint["ema_err"] if "ema_err" in checkpoint else 100
    ema_state = "ema_state" if "ema_state" in checkpoint else "model_state"
    if model_ema:
        logger.info(unwrap_model(model).load_state_dict(checkpoint["model_state"], strict=False))
        unwrap_model(model_ema).load_state_dict(checkpoint[ema_state], strict=False)
    else:
        best_state = "model_state" if test_err <= ema_err else ema_state
        logger.info(unwrap_model(model).load_state_dict(checkpoint[best_state], strict=False))
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint["epoch"], test_err, ema_err


def delete_checkpoints(checkpoint_dir=None, keep="all"):
    assert keep in ["all", "last", "none"], "Invalid keep setting: {}".format(keep)
    checkpoint_dir = checkpoint_dir if checkpoint_dir else get_checkpoint_dir()
    if keep == "all" or not pathmgr.exists(checkpoint_dir):
        return 0
    checkpoints = [f for f in pathmgr.ls(checkpoint_dir) if _NAME_PREFIX in f]
    checkpoints = sorted(checkpoints)[:-1] if keep == "last" else checkpoints
    for checkpoint in checkpoints:
        pathmgr.rm(os.path.join(checkpoint_dir, checkpoint))
    return len(checkpoints)

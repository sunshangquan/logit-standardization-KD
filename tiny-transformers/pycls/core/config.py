#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Modified from pycls.
https://github.com/facebookresearch/pycls/blob/main/pycls/core/config.py
"""

import os

from pycls.core.io import pathmgr
from yacs.config import CfgNode


# Global config object (example usage: from core.config import cfg)
_C = CfgNode()
cfg = _C


# -------------------------- Knowledge distillation options -------------------------- #
_C.DISTILLATION = CfgNode()

# Intermediate layers distillation options
_C.DISTILLATION.ENABLE_INTER = False
_C.DISTILLATION.INTER_TRANSFORM = "linear"
_C.DISTILLATION.INTER_LOSS = "l2"
_C.DISTILLATION.INTER_TEACHER_INDEX = []
_C.DISTILLATION.INTER_STUDENT_INDEX = []
_C.DISTILLATION.INTER_WEIGHT = 2.5

# Logits distillation options
_C.DISTILLATION.ENABLE_LOGIT = False
_C.DISTILLATION.LOGIT_LOSS = "soft"
_C.DISTILLATION.LOGIT_TEMP = 1.0
_C.DISTILLATION.LOGIT_WEIGHT = 0.5

# Teacher model
_C.DISTILLATION.TEACHER_MODEL = "ResNet"
_C.DISTILLATION.TEACHER_WEIGHTS = None
_C.DISTILLATION.TEACHER_IMG_SIZE = 32

# Offline settings
_C.DISTILLATION.OFFLINE = False
_C.DISTILLATION.FEATURE_FILE = None


# ------------------------------- Common model options ------------------------------- #
_C.MODEL = CfgNode()

_C.MODEL.TYPE = "ResNet"
_C.MODEL.IMG_SIZE = 224
_C.MODEL.IN_CHANNELS = 3
_C.MODEL.NUM_CLASSES = 100
_C.MODEL.LOSS_FUN = "cross_entropy"


# ------------------------------------ CNN options ----------------------------------- #
_C.CNN = CfgNode()

_C.CNN.DEPTH = 56
_C.CNN.ACTIVATION_FUN = "relu"
_C.CNN.ACTIVATION_INPLACE = True
_C.CNN.BN_EPS = 1e-5
_C.CNN.BN_MOMENTUM = 0.1
_C.CNN.ZERO_INIT_FINAL_BN_GAMMA = False


_C.RESNET = CfgNode()

_C.RESNET.TRANS_FUN = "basic_transform"
_C.RESNET.NUM_GROUPS = 1
_C.RESNET.WIDTH_PER_GROUP = 64
_C.RESNET.STRIDE_1X1 = True


# -------------------------------- Transformer options ------------------------------- #
_C.TRANSFORMER = CfgNode()

_C.TRANSFORMER.PATCH_SIZE = None
_C.TRANSFORMER.PATCH_STRIDE = None
_C.TRANSFORMER.PATCH_PADDING = None
_C.TRANSFORMER.HIDDEN_DIM = None
_C.TRANSFORMER.DEPTH = None
_C.TRANSFORMER.NUM_HEADS = None
_C.TRANSFORMER.MLP_RATIO = None

_C.TRANSFORMER.LN_EPS = 1e-6
_C.TRANSFORMER.DROP_RATE = None
_C.TRANSFORMER.DROP_PATH_RATE = None
_C.TRANSFORMER.ATTENTION_DROP_RATE = None


_C.T2T = CfgNode()

_C.T2T.TOKEN_DIM = 64
_C.T2T.KERNEL_SIZE = (7, 3, 3)
_C.T2T.STRIDE = (4, 2, 2)
_C.T2T.PADDING = (2, 1, 1)


_C.PIT = CfgNode()

_C.PIT.STRIDE = 8


_C.PVT = CfgNode()

_C.PVT.SR_RATIO = [8, 4, 2, 1]


_C.CONVIT = CfgNode()

_C.CONVIT.LOCAL_LAYERS = 10
_C.CONVIT.LOCALITY_STRENGTH = 1.0


_C.CVT = CfgNode()

_C.CVT.WITH_CLS_TOKEN = [False, False, True]
_C.CVT.QKV_PROJ_METHOD = ['dw_bn', 'dw_bn', 'dw_bn']
_C.CVT.KERNEL_QKV = [3, 3, 3]
_C.CVT.STRIDE_KV = [2, 2, 2]
_C.CVT.STRIDE_Q = [1, 1, 1]
_C.CVT.PADDING_KV = [1, 1, 1]
_C.CVT.PADDING_Q = [1, 1, 1]


# -------------------------------- Optimizer options --------------------------------- #
_C.OPTIM = CfgNode()

# Type of optimizer select from {'sgd', 'adam', 'adamw'}
_C.OPTIM.OPTIMIZER = "sgd"

# Learning rate of body ranges from BASE_LR to MIN_LR according to the LR_POLICY
_C.OPTIM.BASE_LR = 0.1
_C.OPTIM.MIN_LR = 0.0

# Base learning of head is TRANSFER_LR_RATIO * BASE_LR
_C.OPTIM.HEAD_LR_RATIO = 1.0

# Learning rate policy select from {'cos', 'exp', 'lin', 'steps'}
_C.OPTIM.LR_POLICY = "cos"

# Steps for 'steps' policy (in epochs)
_C.OPTIM.STEPS = []

# Learning rate multiplier for 'steps' policy
_C.OPTIM.LR_MULT = 0.1

# Maximal number of epochs
_C.OPTIM.MAX_EPOCH = 200

# Momentum
_C.OPTIM.MOMENTUM = 0.9

# Momentum dampening
_C.OPTIM.DAMPENING = 0.0

# Nesterov momentum
_C.OPTIM.NESTEROV = True

# Betas (for Adam/AdamW optimizer)
_C.OPTIM.BETA1 = 0.9
_C.OPTIM.BETA2 = 0.999

# L2 regularization
_C.OPTIM.WEIGHT_DECAY = 5e-4

# Start the warm up from OPTIM.BASE_LR * OPTIM.WARMUP_FACTOR
_C.OPTIM.WARMUP_FACTOR = 0.1

# Gradually warm up the OPTIM.BASE_LR over this number of epochs
_C.OPTIM.WARMUP_EPOCHS = 0

# Exponential Moving Average (EMA) update value
_C.OPTIM.EMA_ALPHA = 1e-5

# Iteration frequency with which to update EMA weights
_C.OPTIM.EMA_UPDATE_PERIOD = 0


# --------------------------------- Training options --------------------------------- #
_C.TRAIN = CfgNode()

# Dataset and split
_C.TRAIN.DATASET = ""
_C.TRAIN.SPLIT = "train"

# Total mini-batch size
_C.TRAIN.BATCH_SIZE = 128

# Resume training from the latest checkpoint in the output directory
_C.TRAIN.AUTO_RESUME = True

# Weights to start training from
_C.TRAIN.WEIGHTS = ""

# If True train using mixed precision
_C.TRAIN.MIXED_PRECISION = False

# Label smoothing value in 0 to 1 where (0 gives no smoothing)
_C.TRAIN.LABEL_SMOOTHING = 0.0

# Batch mixup regularization value in 0 to 1 (0 gives no mixup)
_C.TRAIN.MIXUP_ALPHA = 0.0

# Batch cutmix regularization value in 0 to 1 (0 gives no cutmix)
_C.TRAIN.CUTMIX_ALPHA = 0.0

_C.TRAIN.STRONG_AUGMENTATION = True 


# --------------------------------- Testing options ---------------------------------- #
_C.TEST = CfgNode()

# Dataset and split
_C.TEST.DATASET = ""
_C.TEST.SPLIT = "val"

# Total mini-batch size
_C.TEST.BATCH_SIZE = 200

# Weights to use for testing
_C.TEST.WEIGHTS = ""


# ------------------------------- Data loader options -------------------------------- #
_C.DATA_LOADER = CfgNode()

# Number of data loader workers per process
_C.DATA_LOADER.NUM_WORKERS = 8

# Load data to pinned host memory
_C.DATA_LOADER.PIN_MEMORY = True


# ---------------------------------- CUDNN options ----------------------------------- #
_C.CUDNN = CfgNode()

# Perform benchmarking to select fastest CUDNN algorithms (best for fixed input sizes)
_C.CUDNN.BENCHMARK = True


# ------------------------------- Precise time options ------------------------------- #
_C.PREC_TIME = CfgNode()

# Number of iterations to warm up the caches
_C.PREC_TIME.WARMUP_ITER = 3

# Number of iterations to compute avg time
_C.PREC_TIME.NUM_ITER = 30


# ---------------------------------- Launch options ---------------------------------- #
_C.LAUNCH = CfgNode()

# The launch mode, may be 'local' or 'slurm' (or 'submitit_local' for debugging)
# The 'local' mode uses a multi-GPU setup via torch.multiprocessing.run_processes.
# The 'slurm' mode uses submitit to launch a job on a SLURM cluster and provides
# support for MULTI-NODE jobs (and is the only way to launch MULTI-NODE jobs).
# In 'slurm' mode, the LAUNCH options below can be used to control the SLURM options.
# Note that NUM_GPUS (not part of LAUNCH options) determines total GPUs requested.
_C.LAUNCH.MODE = "local"

# Launch options that are only used if LAUNCH.MODE is 'slurm'
_C.LAUNCH.MAX_RETRY = 3
_C.LAUNCH.NAME = "pycls_job"
_C.LAUNCH.COMMENT = ""
_C.LAUNCH.CPUS_PER_GPU = 10
_C.LAUNCH.MEM_PER_GPU = 60
_C.LAUNCH.PARTITION = "devlab"
_C.LAUNCH.GPU_TYPE = "volta"
_C.LAUNCH.TIME_LIMIT = 4200
_C.LAUNCH.EMAIL = ""


# ----------------------------------- Misc options ----------------------------------- #
# Optional description of a config
_C.DESC = ""

# If True output additional info to log
_C.VERBOSE = True

# Number of GPUs to use (applies to both training and testing)
_C.NUM_GPUS = 1

# Maximum number of GPUs available per node (unlikely to need to be changed)
_C.MAX_GPUS_PER_NODE = 8

# Output directory
_C.OUT_DIR = None

# Config destination (in OUT_DIR)
_C.CFG_DEST = "config.yaml"

# Note that non-determinism is still be present due to non-deterministic GPU ops
_C.RNG_SEED = 1

# Log destination ('stdout' or 'file')
_C.LOG_DEST = "stdout"

# Log period in iters
_C.LOG_PERIOD = 10

# Distributed backend
_C.DIST_BACKEND = "nccl"

# Hostname and port range for multi-process groups (actual port selected randomly)
_C.HOST = "localhost"
_C.PORT_RANGE = [10000, 65000]

# Models weights referred to by URL are downloaded to this local cache
_C.DOWNLOAD_CACHE = "/tmp/pycls-download-cache"


# ---------------------------------- Default config ---------------------------------- #
_CFG_DEFAULT = _C.clone()
_CFG_DEFAULT.freeze()


def assert_cfg():
    """Checks config values invariants."""
    err_str = "The first lr step must start at 0"
    assert not _C.OPTIM.STEPS or _C.OPTIM.STEPS[0] == 0, err_str
    data_splits = ["train", "val", "test"]
    err_str = "Data split '{}' not supported"
    assert _C.TRAIN.SPLIT in data_splits, err_str.format(_C.TRAIN.SPLIT)
    assert _C.TEST.SPLIT in data_splits, err_str.format(_C.TEST.SPLIT)
    err_str = "Mini-batch size should be a multiple of NUM_GPUS."
    assert _C.TRAIN.BATCH_SIZE % _C.NUM_GPUS == 0, err_str
    assert _C.TEST.BATCH_SIZE % _C.NUM_GPUS == 0, err_str
    err_str = "Log destination '{}' not supported"
    assert _C.LOG_DEST in ["stdout", "file"], err_str.format(_C.LOG_DEST)
    err_str = "NUM_GPUS must be divisible by or less than MAX_GPUS_PER_NODE"
    num_gpus, max_gpus_per_node = _C.NUM_GPUS, _C.MAX_GPUS_PER_NODE
    assert num_gpus <= max_gpus_per_node or num_gpus % max_gpus_per_node == 0, err_str
    err_str = "Invalid mode {}".format(_C.LAUNCH.MODE)
    assert _C.LAUNCH.MODE in ["local", "submitit_local", "slurm"], err_str


def dump_cfg():
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(_C.OUT_DIR, _C.CFG_DEST)
    with pathmgr.open(cfg_file, "w") as f:
        _C.dump(stream=f)
    return cfg_file


def load_cfg(cfg_file):
    """Loads config from specified file."""
    with pathmgr.open(cfg_file, "r") as f:
        _C.merge_from_other_cfg(_C.load_cfg(f))


def reset_cfg():
    """Reset config to initial state."""
    _C.merge_from_other_cfg(_CFG_DEFAULT)

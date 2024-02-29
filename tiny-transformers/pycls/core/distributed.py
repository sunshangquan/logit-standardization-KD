#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Modified from pycls.
https://github.com/facebookresearch/pycls/blob/main/pycls/core/distributed.py
"""

import os
import random

import submitit
import torch
from pycls.core.config import cfg


os.environ["MKL_THREADING_LAYER"] = "GNU"


class SubmititRunner(submitit.helpers.Checkpointable):

    def __init__(self, port, fun, cfg_state):
        self.cfg_state = cfg_state
        self.port = port
        self.fun = fun

    def __call__(self):
        job_env = submitit.JobEnvironment()
        os.environ["MASTER_ADDR"] = job_env.hostnames[0]
        os.environ["MASTER_PORT"] = str(self.port)
        os.environ["RANK"] = str(job_env.global_rank)
        os.environ["LOCAL_RANK"] = str(job_env.local_rank)
        os.environ["WORLD_SIZE"] = str(job_env.num_tasks)
        setup_distributed(self.cfg_state)
        self.fun()


def is_main_proc(local=False):
    m = cfg.MAX_GPUS_PER_NODE if local else cfg.NUM_GPUS
    return cfg.NUM_GPUS == 1 or torch.distributed.get_rank() % m == 0


def scaled_all_reduce(tensors):
    if cfg.NUM_GPUS == 1:
        return tensors
    reductions = []
    for tensor in tensors:
        reduction = torch.distributed.all_reduce(tensor, async_op=True)
        reductions.append(reduction)
    for reduction in reductions:
        reduction.wait()
    for tensor in tensors:
        tensor.mul_(1.0 / cfg.NUM_GPUS)
    return tensors


def setup_distributed(cfg_state):
    cfg.defrost()
    cfg.update(**cfg_state)
    cfg.freeze()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.distributed.init_process_group(backend=cfg.DIST_BACKEND)
    torch.cuda.set_device(local_rank)


def single_proc_run(local_rank, fun, main_port, cfg_state, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(main_port)
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    setup_distributed(cfg_state)
    fun()


def multi_proc_run(num_proc, fun):
    launch = cfg.LAUNCH
    if launch.MODE in ["submitit_local", "slurm"]:
        use_slurm = launch.MODE == "slurm"
        executor = submitit.AutoExecutor if use_slurm else submitit.LocalExecutor
        kwargs = {"slurm_max_num_timeout": launch.MAX_RETRY} if use_slurm else {}
        executor = executor(folder=cfg.OUT_DIR, **kwargs)
        num_gpus_per_node = min(cfg.NUM_GPUS, cfg.MAX_GPUS_PER_NODE)
        executor.update_parameters(
            mem_gb=launch.MEM_PER_GPU * num_gpus_per_node,
            gpus_per_node=num_gpus_per_node,
            tasks_per_node=num_gpus_per_node,
            cpus_per_task=launch.CPUS_PER_GPU,
            nodes=max(1, cfg.NUM_GPUS // cfg.MAX_GPUS_PER_NODE),
            timeout_min=launch.TIME_LIMIT,
            name=launch.NAME,
            slurm_partition=launch.PARTITION,
            slurm_comment=launch.COMMENT,
            slurm_constraint=launch.GPU_TYPE,
            slurm_additional_parameters={"mail-user": launch.EMAIL, "mail-type": "END"},
        )
        main_port = random.randint(cfg.PORT_RANGE[0], cfg.PORT_RANGE[1])
        job = executor.submit(SubmititRunner(main_port, fun, cfg))
        print("Submitted job_id {} with out_dir: {}".format(job.job_id, cfg.OUT_DIR))
        if not use_slurm:
            job.wait()
    elif num_proc > 1:
        main_port = random.randint(cfg.PORT_RANGE[0], cfg.PORT_RANGE[1])
        mp_runner = torch.multiprocessing.start_processes
        args = (fun, main_port, cfg, num_proc)
        mp_runner(single_proc_run, args=args, nprocs=num_proc, start_method="fork")
    else:
        fun()

# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

# from https://github.com/FoundationVision/LlamaGen/blob/main/utils/distributed.py
import os
import sys
import glob
import torch
import subprocess
import torch.distributed as dist
import datetime
import logging

from infinity.models.videovae.utils.misc import rank_zero_only, COLOR_BLUE, COLOR_RESET

from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
)
from infinity.models.videovae.models.cvivit_vqgan import CViViT_Decoder, CViViT_Encoder


def setup_for_distributed(is_master, logging_dir=""):
    """
    This function disables printing when not in master process and 
    redirects stdout to log_out.txt and stderr to log_err.txt.
    """
    import builtins as __builtin__

    class Logger(logging.StreamHandler):
        def __init__(self, stream, file):
            super().__init__(stream)
            self.file = file

        def emit(self, record):
            try:
                msg = self.format(record)
                stream = self.stream
                fs = "%s\n"

                # Stream to the original stream and then flush
                stream.write(fs % msg)
                stream.flush()

                # Stream to the file and then flush
                self.file.write(fs % msg)
                self.file.flush()
            except Exception as e:
                self.handleError(record)

        def isatty(self):
            # Mimic the isatty method usually found in file-like objects
            return self.stream.isatty()

    # print rank 0 only
    builtin_print = __builtin__.print
    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    __builtin__.print = print

    if is_master:
        os.makedirs(logging_dir, exist_ok=True)
        existing_logs = glob.glob(os.path.join(logging_dir, 'log_out_*.txt'))
        log_numbers = [int(log.split('.txt')[0].split('_')[-1]) for log in existing_logs]
        next_log_number = max(log_numbers) + 1 if log_numbers else 1
        
        log_out_path = os.path.join(logging_dir, f'log_out_{next_log_number}.txt')
        log_err_path = os.path.join(logging_dir, f'log_err_{next_log_number}.txt')
    
        logger_stdout = Logger(sys.stdout, open(log_out_path, 'w'))
        logger_stderr = Logger(sys.stderr, open(log_err_path, 'w'))
        logging.basicConfig(level=logging.DEBUG, handlers=[logger_stdout, logger_stderr])

        print(f"{COLOR_BLUE}stdout will be written to {log_out_path}{COLOR_RESET}")
        print(f"{COLOR_BLUE}stderr will be written to {log_err_path}{COLOR_RESET}")

def init_distributed_mode(args, timeout_minutes=15):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.dist_url = 'env://'
        os.environ['LOCAL_SIZE'] = str(torch.cuda.device_count())
    elif 'SLURM_PROCID' in os.environ:
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(
            'scontrol show hostname {} | head -n1'.format(node_list))
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
        os.environ['LOCAL_SIZE'] = str(num_gpus)
        args.dist_url = 'env://'
        args.world_size = ntasks
        args.rank = proc_id
        args.gpu = proc_id % num_gpus
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank,
                                         timeout=datetime.timedelta(seconds=timeout_minutes * 60)
                                         )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0, args.default_root_dir)

def _FSDP(model: torch.nn.Module, device, zero) -> FSDP:
    model = FSDP(
        model,
        auto_wrap_policy=ModuleWrapPolicy([CViViT_Encoder, CViViT_Decoder]),
        device_id=device,
        sharding_strategy={1:ShardingStrategy.HYBRID_SHARD, 2:ShardingStrategy.SHARD_GRAD_OP, 3:ShardingStrategy.FULL_SHARD}.get(zero),
        mixed_precision=MixedPrecision(
            param_dtype=torch.float,
            reduce_dtype=torch.float,
            buffer_dtype=torch.float,
        ),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,
    )
    torch.cuda.synchronize()
    return model


def reduce_losses(loss_dict, dst=0):
    loss_names = list(loss_dict.keys())
    loss_tensor = torch.stack([loss_dict[name] for name in loss_names])

    dist.reduce(loss_tensor, dst=dst, op=dist.ReduceOp.SUM)
    # Only average the loss values on the destination rank
    if dist.get_rank() == dst:
        loss_tensor /= dist.get_world_size()
        averaged_losses = {name: loss_tensor[i].item() for i, name in enumerate(loss_names)}
    else:
        averaged_losses = {name: None for name in loss_names}
    
    return averaged_losses

@rank_zero_only
def average_losses(loss_dict_list):
    sum_dict = {}
    count_dict = {}
    for loss_dict in loss_dict_list:
        for key, value in loss_dict.items():
            if key in sum_dict:
                sum_dict[key] += value
                count_dict[key] += 1
            else:
                sum_dict[key] = value
                count_dict[key] = 1

    avg_dict = {key: sum_dict[key] / count_dict[key] for key in sum_dict}
    return avg_dict

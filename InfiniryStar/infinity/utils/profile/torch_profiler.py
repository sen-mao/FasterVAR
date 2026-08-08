# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
import logging
import os
from contextlib import contextmanager, nullcontext
from datetime import datetime

import torch
import torch.distributed as dist
from torch.profiler import record_function as torch_record_function


class _TraceHandler:
    def __init__(self, save_path="/tmp/trace.json", logger=None, rank=None):
        self.logger = logger
        if logger is None:
            self.logger = logging.getLogger(__name__)

        self.logger.info(f"trace dump path: {save_path}")
        self.save_path = save_path + ".json.gz"
        self.rank = rank

    def __call__(self, prof):
        if self.logger is not None:
            self.logger.info(f"dump trace to {self.save_path}")
        prof.export_chrome_trace(self.save_path)

class torch_profiler:
    """
    usage:

    ```python
    import pnp

    pnp.torch_profiler.setup(output_folder="./", wait_steps=30)

    for step in range(100):
        pnp.torch_profiler.step()
        ...
    
        with pnp.troch_profiler.mark("fwd"):
            model.forward()

        ...

        with pnp.torch_profiler.mark("bwd"):
            loss.backward()

    ```

    """
    _TP = None
    mark = nullcontext

    @staticmethod
    def step():
        if torch_profiler._TP is None:
            return

        torch_profiler._TP.step()

    @staticmethod
    @property
    def mark():
        return torch_profiler.mark

    @staticmethod
    def setup(enabled=True, output_folder="./", file_prefix="", wait_steps=30):
        """
        enabled: if False, profiler will do nothing
        output_folder: the folder to dump trace
        wait_steps: start profiling after wait_steps(in your training loop)
        file_prefix: the prefix of the trace file for your custom
        """
        if enabled:
           if not os.path.exists(output_folder):
               os.makedirs(output_folder, exist_ok=True)
  
           torch_profiler._TP = torch.profiler.profile(
               activities=[
                   torch.profiler.ProfilerActivity.CPU,
                   torch.profiler.ProfilerActivity.CUDA,
               ],
               schedule=torch.profiler.schedule(
                   wait=wait_steps,
                   warmup=3,
                   active=5,
                   repeat=0,
               ),
               with_stack=True,
               record_shapes=True,
               profile_memory=False,
               on_trace_ready=_TraceHandler(
                   f"{output_folder}/{file_prefix}world_size-{dist.get_world_size()}-rank{dist.get_rank()}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                   None,
                   dist.get_rank(),
               ),
           )
           torch_profiler._TP.start()
           torch_profiler.mark = torch_record_function

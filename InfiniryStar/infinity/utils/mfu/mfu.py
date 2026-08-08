# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
import time
import torch
import torch.distributed as dist
from contextlib import contextmanager, nullcontext
from functools import wraps
from .flops_profiler import FlopsProfiler
from .flops_calc_impl.custom_flops_impl import CUSTOM_HOOK_MAPPING, CUSTOM_NAME_MAPPING

class _MFU:
    def __init__(self, calibration_steps = 5, repeat_after_steps = -1):
        """
        calibration_steps = -1 means always do calibration, has a very little overhead
        repeat_after_steps = -1 means never repeat
        """
        self.profs = []
        self.iter_time = None
        self.is_during_calibration = False
        self.calibration_steps = calibration_steps
        self.repeat_after_steps = repeat_after_steps
        self.steps = 0
        self.flops = []
        self.detail_flops = ""
        self.ideal_TFLOPS = self._get_device_tflops()
        self.ignore_list=[]
        self.prof = FlopsProfiler()

    def append(self, model):
        self.prof.append(model)

    def step(self, iter_time):
        self.steps += 1
        self.iter_time = iter_time

        if self.calibration_steps < 0 or self.steps <= self.calibration_steps:
            self.is_during_calibration = True
            flop = 0

            try:
                flop, log = self.prof.get_total_flops()
            except Exception as e:
                print(f"[WARN]: get_total_flops failed {e}")
            
            self.detail_flops = log
            self.flops.append(flop)
            self.reset()

            if self.steps == self.calibration_steps:
                self.is_during_calibration = False
                self.clear()

        if self.calibration_steps > 0 and self.repeat_after_steps > 0:
            if self.steps >= self.calibration_steps + self.repeat_after_steps:
                self.flops.clear()
                self.steps = 0
                self.start()


    def stop(self):
       self.prof.stop_profile()

    def reset(self):
       self.prof.reset_profile()

    def clear(self):
        self.prof.end_profile()

    def start(self):
        self.prof.start_profile(self.ignore_list)

    def get_flops_detail_info(self):
        return self.detail_flops

    def get_mfu(self):
        mfu = -1
        if self.iter_time is not None and len(self.flops) > 0:
            avg_flop = sum(self.flops) / len(self.flops)
            avg_Tflops = avg_flop / 1e12
            mfu = avg_Tflops / self.iter_time / self.ideal_TFLOPS
            if not isinstance(mfu, float):
                print(f"[WARN]: Something wrong with mfu calc, {type(mfu)=}.")
                mfu = -1

        return mfu

    def _get_device_tflops(self):
        peak_tflops = -1
        arch = torch.cuda.get_device_capability()
        if arch[0] == 8 and arch[1] == 0:  # A100/A800
            peak_tflops = 312 # fp16 without sparsity
        elif arch[0] == 9 and arch[1] == 0:  # H100/H800
            peak_tflops = 989 # fp16 without sparsity
        else:
            print(f"unknown default tflops of device capability {arch[0]}.{arch[1]}")
        return peak_tflops



class mfutool:
    _mfu = None
    _last_time = None
    _iter_time = None

    @staticmethod
    def setup(calibration_steps = 5, repeat_after_steps = -1):
        """
        calibration_steps = -1 means always do calibration, has a very little overhead
        repeat_after_steps = -1 means never repeat
        """
        if mfutool._mfu is None:
            mfutool._mfu = _MFU(calibration_steps = calibration_steps, repeat_after_steps = repeat_after_steps)

    @staticmethod
    def add(model):
        if mfutool._mfu is None:
            mfutool._mfu = _MFU()
        mfutool._mfu.append(model)

    @staticmethod
    def enable():
        if mfutool._mfu is not None:
            mfutool._mfu.start()

    @staticmethod
    def disable():
        if mfutool._mfu is not None:
            mfutool._mfu.stop()

    @staticmethod
    def step():
        if mfutool._mfu is not None:
            if mfutool._last_time is not None:
                mfutool._iter_time = time.time() - mfutool._last_time
                mfutool._mfu.step(mfutool._iter_time)
            mfutool._last_time = time.time()

    @staticmethod
    def iter_time():
        return mfutool._iter_time

    @staticmethod
    def get_mfu():
        if mfutool._mfu is not None:
            return mfutool._mfu.get_mfu()

    @staticmethod
    def get_flops_detail_info():
        if mfutool._mfu is not None:
            return mfutool._mfu.get_flops_detail_info()

    @staticmethod
    def register_custom(name, func):
        if name not in CUSTOM_NAME_MAPPING:
            print(f"[WARN] cannot find {name}, decorate your module class with @mfutool.custom_flops first")
            return
        CUSTOM_HOOK_MAPPING[CUSTOM_NAME_MAPPING[name]] = func
    
    @staticmethod
    def custom_flops(cls, name):
        CUSTOM_NAME_MAPPING[name] = cls
        return cls


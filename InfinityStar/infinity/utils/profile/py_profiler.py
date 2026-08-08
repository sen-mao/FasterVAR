# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
import logging
import pstats
import cProfile
import contextlib

def _colored(st, color, background=False):
    return f"\u001b[{10*background+60*(color.upper() == color)+30+['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'].index(color.lower())}m{st}\u001b[0m" if color is not None else st 

def _format_fcn(fcn):
    return f"{fcn[0]}:{fcn[1]}:{fcn[2]}"

class py_profiler(contextlib.ContextDecorator):
    def __init__(self, enabled=True, sort='cumtime', fn=None, ts=1):
        self.enabled, self.sort, self.fn, self.time_scale = enabled, sort, fn, 1e3/ts
    def __enter__(self):
        self.pr = cProfile.Profile()
        if self.enabled:
            self.pr.enable()
    def __exit__(self, *exc):
        if self.enabled:
            self.pr.disable()
            if self.fn:
                self.pr.dump_stats(self.fn)
            stats = pstats.Stats(self.pr).strip_dirs().sort_stats(self.sort)
            for fcn in stats.fcn_list[0:int(len(stats.fcn_list))]:
                (_primitive_calls, num_calls, tottime, cumtime, callers) = stats.stats[fcn]
                scallers = sorted(callers.items(), key=lambda x: -x[1][2])
                print(f"n:{num_calls:8d}  tm:{tottime*self.time_scale:7.2f}ms  tot:{cumtime*self.time_scale:7.2f}ms", _colored(_format_fcn(fcn).ljust(50), "yellow"))


if __name__ == "__main__":
    def fn():
        s = 0
        for i in range(10000000):
            s += i
        return s

    with py_profiler():
        fn()

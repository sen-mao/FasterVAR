# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

# source: https://github.com/microsoft/DeepSpeed/tree/master/deepspeed/profiling/flops_profiler

# DeepSpeed Team
import os
import time
import torch
import torch.nn.functional as F
import logging
from functools import partial
import einops

from .flops_calc_impl.func_flops_impl import *
from .flops_calc_impl.nn_flops_impl import *
from .flops_calc_impl.tensor_flops_impl import *
from .flops_calc_impl.custom_flops_impl import *

logger = logging.getLogger(__name__)

old_functions = {}

DEFAULT_PRECISION = 2

class FlopsProfiler(object):
    """Measures the latency, number of estimated floating-point operations and parameters of each module in a PyTorch model.

    The flops-profiler profiles the forward pass of a PyTorch model and prints the model graph with the measured profile attached to each module. It shows how latency, flops and parameters are spent in the model and which modules or layers could be the bottleneck. It also outputs the names of the top k modules in terms of aggregated latency, flops, and parameters at depth l with k and l specified by the user. The output profile is computed for each batch of input.
    The DeepSpeed flops profiler can be used with the DeepSpeed runtime or as a standalone package.
    When using DeepSpeed for model training, the flops profiler can be configured in the deepspeed_config file and no user code change is required.

    If using the profiler as a standalone package, one imports the flops_profiler package and use the APIs.

    Here is an example for usage in a typical training workflow:

        .. code-block:: python

            model = Model()
            prof = FlopsProfiler(model)

            for step, batch in enumerate(data_loader):
                if step == profile_step:
                    prof.start_profile()

                loss = model(batch)

                if step == profile_step:
                    flops = prof.get_total_flops()
                    prof.end_profile()

                loss.backward()
                optimizer.step()

    To profile a trained model in inference, use the `get_model_profile` API.

    Args:
        object (torch.nn.Module): The PyTorch model to profile.
    """

    def __init__(self):
        self.models = []
        self.started = False
        self.func_patched = False
        self.module_flop_count = []
        self.detail_flops = ""

    def append(self, model):
        self.models.append(model)

    def start_profile(self, ignore_list=None):
        """Starts profiling.

        Extra attributes are added recursively to all the modules and the profiled torch.nn.functionals are monkey patched.

        Args:
            ignore_list (list, optional): the list of modules to ignore while profiling. Defaults to None.
        """
        self.ignore_list = ignore_list
        self.reset_profile()
        _patch_functionals(self.module_flop_count)
        _patch_tensor_methods(self.module_flop_count)
        _patch_miscellaneous_operations(self.module_flop_count)

        def register_module_hooks(module, ignore_list):
            if ignore_list and type(module) in ignore_list:
                return

            # if computing the flops of a module directly
            if type(module) in MODULE_HOOK_MAPPING:
                if not hasattr(module, "__flops_handle__"):
                    module.__flops_handle__ = module.register_forward_hook(MODULE_HOOK_MAPPING[type(module)])
                return

            if type(module) in CUSTOM_HOOK_MAPPING:
                if not hasattr(module, "__flops_handle__"):
                    module.__flops_handle__ = module.register_forward_hook(CUSTOM_HOOK_MAPPING[type(module)], with_kwargs=True)
                return

            # if computing the flops of the functionals in a module
            def pre_hook(module, input):
                self.module_flop_count.append([])

            if not hasattr(module, "__pre_hook_handle__"):
                module.__pre_hook_handle__ = module.register_forward_pre_hook(pre_hook)

            def post_hook(module, input, output):
                if self.module_flop_count:

                    if torch.is_grad_enabled():
                        module.__flops__ += sum([elem[1] for elem in self.module_flop_count[-1]]) * (3 if module.training else 1)

                    self.module_flop_count.pop()

            if not hasattr(module, "__post_hook_handle__"):
                module.__post_hook_handle__ = module.register_forward_hook(post_hook)


        for model in self.models:
            model.apply(partial(register_module_hooks, ignore_list=ignore_list))

        self.started = True
        self.func_patched = True
        logger.info("Flops profiler started")

    def stop_profile(self):
        """Stop profiling.

        All torch.nn.functionals are restored to their originals.
        """
        self.module_flop_count.clear()
        if self.started and self.func_patched:
            _reload_functionals()
            _reload_tensor_methods()
            _reload_miscellaneous_operations()
            self.func_patched = False

        def remove_profile_attrs(module):
            if hasattr(module, "__pre_hook_handle__"):
                module.__pre_hook_handle__.remove()
                del module.__pre_hook_handle__
            if hasattr(module, "__post_hook_handle__"):
                module.__post_hook_handle__.remove()
                del module.__post_hook_handle__
            if hasattr(module, "__flops_handle__"):
                module.__flops_handle__.remove()
                del module.__flops_handle__

        for model in self.models:
            model.apply(remove_profile_attrs)

    def reset_profile(self):
        """Resets the profiling.

        Adds or resets the extra attributes.
        """
        self.module_flop_count.clear()
        def add_or_reset_attrs(module):
            module.__flops__ = 0

        for model in self.models:
            model.apply(add_or_reset_attrs)

    def end_profile(self):
        """Ends profiling.

        The added attributes and handles are removed recursively on all the modules.
        """
        if not self.started:
            return
        self.stop_profile()
        self.started = False
        self.module_flop_count.clear()

        def remove_profile_attrs(module):
            if hasattr(module, "__flops__"):
                del module.__flops__

        for model in self.models:
            model.apply(remove_profile_attrs)
        logger.info("Flops profiler finished")

    def get_total_flops(self):
        """Returns the total flops of the model.

        Returns:
            The number of multiply-accumulate operations of the model forward pass.
        """
        total_flops = 0
        self.detail_flops = ""
        for model in self.models:
            flops, log = get_module_flops(model, prefix="")
            total_flops += flops
            self.detail_flops += log
        return total_flops, self.detail_flops

def wrapFunc(func, funcFlopCompute, module_flop_count):
    oldFunc = func
    name = func.__str__
    old_functions[name] = oldFunc

    @torch.compiler.disable()
    def newFunc(*args, **kwds):
        flops, macs = funcFlopCompute(*args, **kwds)
        if module_flop_count:
            module_flop_count[-1].append((name, flops, func.__name__))
        return oldFunc(*args, **kwds)

    newFunc.__str__ = func.__str__

    return newFunc


def _patch_functionals(module_flop_count):
    # FC
    F.linear = wrapFunc(F.linear, linear_flops_compute, module_flop_count)

    # convolutions
    F.conv1d = wrapFunc(F.conv1d, conv_flops_compute, module_flop_count)
    F.conv2d = wrapFunc(F.conv2d, conv_flops_compute, module_flop_count)
    F.conv3d = wrapFunc(F.conv3d, conv_flops_compute, module_flop_count)

    # conv transposed
    F.conv_transpose1d = wrapFunc(F.conv_transpose1d, conv_trans_flops_compute, module_flop_count)
    F.conv_transpose2d = wrapFunc(F.conv_transpose2d, conv_trans_flops_compute, module_flop_count)
    F.conv_transpose3d = wrapFunc(F.conv_transpose3d, conv_trans_flops_compute, module_flop_count)

    # activations
    F.relu = wrapFunc(F.relu, relu_flops_compute, module_flop_count)
    F.prelu = wrapFunc(F.prelu, prelu_flops_compute, module_flop_count)
    F.elu = wrapFunc(F.elu, elu_flops_compute, module_flop_count)
    F.leaky_relu = wrapFunc(F.leaky_relu, leaky_relu_flops_compute, module_flop_count)
    F.relu6 = wrapFunc(F.relu6, relu6_flops_compute, module_flop_count)
    if hasattr(F, "silu"):
        F.silu = wrapFunc(F.silu, silu_flops_compute, module_flop_count)
    F.gelu = wrapFunc(F.gelu, gelu_flops_compute, module_flop_count)

    # Normalizations
    F.batch_norm = wrapFunc(F.batch_norm, batch_norm_flops_compute, module_flop_count)
    F.layer_norm = wrapFunc(F.layer_norm, layer_norm_flops_compute, module_flop_count)
    F.instance_norm = wrapFunc(F.instance_norm, instance_norm_flops_compute, module_flop_count)
    F.group_norm = wrapFunc(F.group_norm, group_norm_flops_compute, module_flop_count)

    # poolings
    F.avg_pool1d = wrapFunc(F.avg_pool1d, pool_flops_compute, module_flop_count)
    F.avg_pool2d = wrapFunc(F.avg_pool2d, pool_flops_compute, module_flop_count)
    F.avg_pool3d = wrapFunc(F.avg_pool3d, pool_flops_compute, module_flop_count)
    F.max_pool1d = wrapFunc(F.max_pool1d, pool_flops_compute, module_flop_count)
    F.max_pool2d = wrapFunc(F.max_pool2d, pool_flops_compute, module_flop_count)
    F.max_pool3d = wrapFunc(F.max_pool3d, pool_flops_compute, module_flop_count)
    F.adaptive_avg_pool1d = wrapFunc(F.adaptive_avg_pool1d, pool_flops_compute, module_flop_count)
    F.adaptive_avg_pool2d = wrapFunc(F.adaptive_avg_pool2d, pool_flops_compute, module_flop_count)
    F.adaptive_avg_pool3d = wrapFunc(F.adaptive_avg_pool3d, pool_flops_compute, module_flop_count)
    F.adaptive_max_pool1d = wrapFunc(F.adaptive_max_pool1d, pool_flops_compute, module_flop_count)
    F.adaptive_max_pool2d = wrapFunc(F.adaptive_max_pool2d, pool_flops_compute, module_flop_count)
    F.adaptive_max_pool3d = wrapFunc(F.adaptive_max_pool3d, pool_flops_compute, module_flop_count)

    # upsample
    F.upsample = wrapFunc(F.upsample, upsample_flops_compute, module_flop_count)
    F.interpolate = wrapFunc(F.interpolate, upsample_flops_compute, module_flop_count)

    # softmax
    F.softmax = wrapFunc(F.softmax, softmax_flops_compute, module_flop_count)

    # embedding
    F.embedding = wrapFunc(F.embedding, embedding_flops_compute, module_flop_count)

    # attn - scaled_dot_product_attention added in torch 2.0+
    F.scaled_dot_product_attention = wrapFunc(F.scaled_dot_product_attention, attn_flops_compute, module_flop_count)

def _patch_tensor_methods(module_flop_count):
    torch.matmul = wrapFunc(torch.matmul, matmul_flops_compute, module_flop_count)
    torch.Tensor.matmul = wrapFunc(torch.Tensor.matmul, matmul_flops_compute, module_flop_count)
    torch.Tensor.__matmul__ = wrapFunc(torch.Tensor.__matmul__, matmul_flops_compute, module_flop_count)
    torch.mm = wrapFunc(torch.mm, matmul_flops_compute, module_flop_count)
    torch.Tensor.mm = wrapFunc(torch.Tensor.mm, matmul_flops_compute, module_flop_count)
    torch.bmm = wrapFunc(torch.bmm, matmul_flops_compute, module_flop_count)
    torch.Tensor.bmm = wrapFunc(torch.Tensor.bmm, matmul_flops_compute, module_flop_count)

    torch.addmm = wrapFunc(torch.addmm, addmm_flops_compute, module_flop_count)
    torch.Tensor.addmm = wrapFunc(torch.Tensor.addmm, tensor_addmm_flops_compute, module_flop_count)

    torch.mul = wrapFunc(torch.mul, mul_flops_compute, module_flop_count)
    torch.Tensor.mul = wrapFunc(torch.Tensor.mul, mul_flops_compute, module_flop_count)

    torch.add = wrapFunc(torch.add, add_flops_compute, module_flop_count)
    torch.Tensor.add = wrapFunc(torch.Tensor.add, add_flops_compute, module_flop_count)

    torch.einsum = wrapFunc(torch.einsum, einsum_flops_compute, module_flop_count)

    torch.baddbmm = wrapFunc(torch.baddbmm, tensor_addmm_flops_compute, module_flop_count)


def _patch_miscellaneous_operations(module_flop_count):
    einops.einsum = wrapFunc(einops.einsum, einops_einsum_flops_compute, module_flop_count)


def _reload_functionals():
    # torch.nn.functional does not support importlib.reload()
    F.linear = old_functions[F.linear.__str__]
    F.conv1d = old_functions[F.conv1d.__str__]
    F.conv2d = old_functions[F.conv2d.__str__]
    F.conv3d = old_functions[F.conv3d.__str__]
    F.conv_transpose1d = old_functions[F.conv_transpose1d.__str__]
    F.conv_transpose2d = old_functions[F.conv_transpose2d.__str__]
    F.conv_transpose3d = old_functions[F.conv_transpose3d.__str__]
    F.relu = old_functions[F.relu.__str__]
    F.prelu = old_functions[F.prelu.__str__]
    F.elu = old_functions[F.elu.__str__]
    F.leaky_relu = old_functions[F.leaky_relu.__str__]
    F.relu6 = old_functions[F.relu6.__str__]
    if hasattr(F, "silu"):
        F.silu = old_functions[F.silu.__str__]
    F.gelu = old_functions[F.gelu.__str__]
    F.batch_norm = old_functions[F.batch_norm.__str__]
    F.layer_norm = old_functions[F.layer_norm.__str__]
    F.instance_norm = old_functions[F.instance_norm.__str__]
    F.group_norm = old_functions[F.group_norm.__str__]
    F.avg_pool1d = old_functions[F.avg_pool1d.__str__]
    F.avg_pool2d = old_functions[F.avg_pool2d.__str__]
    F.avg_pool3d = old_functions[F.avg_pool3d.__str__]
    F.max_pool1d = old_functions[F.max_pool1d.__str__]
    F.max_pool2d = old_functions[F.max_pool2d.__str__]
    F.max_pool3d = old_functions[F.max_pool3d.__str__]
    F.adaptive_avg_pool1d = old_functions[F.adaptive_avg_pool1d.__str__]
    F.adaptive_avg_pool2d = old_functions[F.adaptive_avg_pool2d.__str__]
    F.adaptive_avg_pool3d = old_functions[F.adaptive_avg_pool3d.__str__]
    F.adaptive_max_pool1d = old_functions[F.adaptive_max_pool1d.__str__]
    F.adaptive_max_pool2d = old_functions[F.adaptive_max_pool2d.__str__]
    F.adaptive_max_pool3d = old_functions[F.adaptive_max_pool3d.__str__]
    F.upsample = old_functions[F.upsample.__str__]
    F.interpolate = old_functions[F.interpolate.__str__]
    F.softmax = old_functions[F.softmax.__str__]
    F.embedding = old_functions[F.embedding.__str__]


def _reload_tensor_methods():
    torch.matmul = old_functions[torch.matmul.__str__]
    torch.Tensor.matmul = old_functions[torch.Tensor.matmul.__str__]
    torch.mm = old_functions[torch.mm.__str__]
    torch.Tensor.mm = old_functions[torch.Tensor.mm.__str__]
    torch.bmm = old_functions[torch.matmul.__str__]
    torch.Tensor.bmm = old_functions[torch.Tensor.bmm.__str__]
    torch.addmm = old_functions[torch.addmm.__str__]
    torch.Tensor.addmm = old_functions[torch.Tensor.addmm.__str__]
    torch.mul = old_functions[torch.mul.__str__]
    torch.Tensor.mul = old_functions[torch.Tensor.mul.__str__]
    torch.add = old_functions[torch.add.__str__]
    torch.Tensor.add = old_functions[torch.Tensor.add.__str__]

    torch.einsum = old_functions[torch.einsum.__str__]

    torch.baddbmm = old_functions[torch.baddbmm.__str__]


def _reload_miscellaneous_operations():
    einops.einsum = old_functions[einops.einsum.__str__]

# can not iterate over all submodules using self.model.modules()
# since modules() returns duplicate modules only once
def get_module_flops(module, prefix=""):
    sum = module.__flops__
    log = ""

    if os.getenv("RANK","0") == "0":
        log = f"| {prefix}{module.__class__} flops = {sum/1e12:.5f} T\n"


    for child in module.children():
        flop,clog = get_module_flops(child, prefix=prefix+"    ")
        sum += flop
        log += clog

    return sum, log

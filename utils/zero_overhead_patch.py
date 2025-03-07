import torch
import deepspeed

from zero_overhead_pinned_memory import zeros_cuda_host_alloc_pinned

def _zeros_cpu_zero_overhead(shape: tuple[int], dtype: torch.dtype, pin_memory: bool) -> torch.Tensor:
    return zeros_cuda_host_alloc_pinned(shape, dtype)


def patch_deepspeed_zero_overhead_pinned_memory():
    print("Apply zero overhead pinned memory deepspeed")
    deepspeed.ops.adam.cpu_adam.zeros_cpu_for_momentums = _zeros_cpu_zero_overhead
    deepspeed.ops.adam.cpu_adam.zeros_cpu_for_variances = _zeros_cpu_zero_overhead
    deepspeed.runtime.zero.stage3.zeros_cpu_for_master_weights = _zeros_cpu_zero_overhead
    deepspeed.runtime.zero.stage3.zeros_cpu_for_master_gradients = _zeros_cpu_zero_overhead
    deepspeed.runtime.zero.stage3.zeros_cpu_for_compute_weights = _zeros_cpu_zero_overhead
    deepspeed.runtime.zero.stage3.zeros_cpu_for_compute_gradients = _zeros_cpu_zero_overhead
import functools

import torch
from transformers import AutoModelForCausalLM
import matplotlib.pyplot as plt

from draw import (
    save_memory_usage_figure_gpu,
    save_optimized_memory_usage_figure_gpu,
    save_optimized_memory_usage_figure_cpu,
    save_cpu_memory_usage_figure_cpu,
)

GiB = 1024**3

"""
This code demonstrates various memory usage calculations (weights, optimizer, activations)
for the LLaMA 3.1 (70B) model under different scenarios (original, optimized, CPU offloading, etc.).
"""

def get_element_size(dtype: torch.dtype) -> int:
    """
    Return the size of the given dtype in bytes.
    
    Args:
        dtype (torch.dtype): Data type for which to retrieve element size.

    Returns:
        int: Size of one element of the given dtype in bytes.
    """
    return torch.zeros(1, dtype=dtype).element_size()

def get_checkpointed_size(batch_size: int, seq_length: int) -> int:
    """
    Compute the activation size if gradient checkpointing is enabled.
    For LLaMA 3 70B:
        num_layers = 80
        hidden_size = 8192

    Returns:
        int: Checkpointed activation size in bytes (without scaling by element_size).
    """
    num_layers = 80
    hidden_size = 8192
    return num_layers * hidden_size * batch_size * seq_length


def main():
    # ----------------------------
    # User-configurable parameters
    # ----------------------------
    batch = 1
    seq_length = 128000

    # Dtype for model parameters
    model_precision = torch.bfloat16
    # Dtype for optimizer
    optimizer_precision = torch.float32

    # --------------------
    # Parameter Calculations
    # --------------------
    model_parameter_size = 70.6 * 10**9

    model_precision_size = get_element_size(model_precision)
    optimizer_precision_size = get_element_size(optimizer_precision)

    dynamic_checkpointed_size = model_precision_size * get_checkpointed_size(batch, seq_length)

    # --------------------
    # CPU (DRAM) usage for weights/gradients/optimizer
    # --------------------
    fp32_weight = optimizer_precision_size * model_parameter_size / GiB
    fp32_grad = optimizer_precision_size * model_parameter_size / GiB
    fp32_optimizer = 2 * optimizer_precision_size * model_parameter_size / GiB

    fp16_weight = model_precision_size * model_parameter_size / GiB
    fp16_grad = model_precision_size * model_parameter_size / GiB

    checkpointed = dynamic_checkpointed_size / GiB
    others = 20  # Some fixed overhead

    save_cpu_memory_usage_figure_cpu(
        "optimized_cpu_70B_DRAM.png",
        fp32_weight,
        fp32_grad,
        fp32_optimizer,
        fp16_weight,
        fp16_grad,
        checkpointed,
        others,
    )


if __name__ == "__main__":
    main()

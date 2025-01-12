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
for the LLaMA 3.1 (8B) model under different scenarios (original, optimized, CPU offloading, etc.).
"""


@functools.lru_cache(maxsize=None)
def get_model_parameter_size(model: torch.nn.Module) -> int:
    """
    Calculate the total number of parameters in a PyTorch model.
    
    Args:
        model (torch.nn.Module): The model for which to calculate the parameter size.

    Returns:
        int: Total number of parameters in the model.
    """
    return sum(param.numel() for param in model.parameters())


@functools.lru_cache(maxsize=None)
def get_activation_usage(model: torch.nn.Module) -> int:
    """
    Run a dummy forward pass through the model and accumulate the size of activations
    in bytes. Returns the total activation usage for a single token (or small batch).
    
    Args:
        model (torch.nn.Module): The model for which to calculate activations usage.

    Returns:
        int: Sum of activation sizes in bytes for the forward pass.
    """
    activation_sizes = []

    def forward_hook(module, input, output):
        """
        Hook to calculate activation size for each module. We handle both single-tensor
        and tuple/list outputs.
        """
        if isinstance(output, torch.Tensor):
            activation_sizes.append(output.numel() * output.element_size())
        elif isinstance(output, (tuple, list)):
            for tensor in output:
                if isinstance(tensor, torch.Tensor):
                    activation_sizes.append(tensor.numel() * tensor.element_size())

    # Register hooks to capture intermediate outputs
    hooks = []
    for submodule in model.modules():
        hooks.append(submodule.register_forward_hook(forward_hook))

    # Dummy input of shape (1, 1) (e.g., 1 token in the sequence)
    dummy_input = torch.zeros((1, 1), dtype=torch.long, device="cuda")
    model.eval()
    with torch.no_grad():
        model(dummy_input)

    # Remove hooks to avoid side effects
    for hook in hooks:
        hook.remove()

    return sum(activation_sizes)


def get_element_size(dtype: torch.dtype) -> int:
    """
    Return the size of the given dtype in bytes.
    
    Args:
        dtype (torch.dtype): Data type for which to retrieve element size.

    Returns:
        int: Size of one element of the given dtype in bytes.
    """
    return torch.zeros(1, dtype=dtype).element_size()


def get_llama3_8B_optimized_activation_size(batch_size: int, seq_length: int) -> int:
    """
    Compute the activation size for LLaMA 3.1 8B in an 'optimized' scenario.
    Values are partly empirical:
        same_part = 667942912
        dynamic_part = 59904

    Returns:
        int: Total optimized activation size in bytes (without scaling by element_size).
    """
    same_part = 667942912
    dynamic_part = 59904
    return same_part + (batch_size * seq_length * dynamic_part)


def get_llama3_8B_cpuoffloading_backward_gradient_and_activation_size(batch_size: int, seq_length: int) -> int:
    """
    Compute the backward gradient + activation size for LLaMA 3.1 8B under CPU offloading.
    Values are partly empirical:
        same_part = 802160649
        dynamic_part = 92992

    Returns:
        int: Total CPU-offloaded backward gradient + activation size in bytes 
             (without scaling by element_size).
    """
    same_part = 802160649
    dynamic_part = 92992
    return same_part + (batch_size * seq_length * dynamic_part)


def get_llama3_8B_checkpointed_size(batch_size: int, seq_length: int) -> int:
    """
    Compute the activation size if gradient checkpointing is enabled.
    For LLaMA 3.1 8B:
        num_layers = 32
        hidden_size = 4096

    Returns:
        int: Checkpointed activation size in bytes (without scaling by element_size).
    """
    num_layers = 32
    hidden_size = 4096
    return num_layers * hidden_size * batch_size * seq_length


def main():
    # ----------------------------
    # User-configurable parameters
    # ----------------------------
    batch = 1
    seq_length = 128000
    model_name = "meta-llama/Llama-3.1-8B-Instruct"

    # Dtype for model parameters
    model_precision = torch.bfloat16
    # Dtype for optimizer
    optimizer_precision = torch.float32

    # --------------------
    # Load the model
    # --------------------
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to("cuda", dtype=model_precision)

    # --------------------
    # Parameter Calculations
    # --------------------
    model_parameter_size = get_model_parameter_size(model)

    model_precision_size = get_element_size(model_precision)
    optimizer_precision_size = get_element_size(optimizer_precision)

    # --------------------
    # Static memory usage
    # --------------------
    # - Model weights (in model_precision)
    static_weight_size = model_parameter_size * model_precision_size
    # - Optimizer states (2x model size in optimizer precision, e.g., Adam has 2 states)
    static_optimizer_size = 2 * model_parameter_size * optimizer_precision_size
    # - Assume peak gradient memory ~ size of the model in model precision
    dynamic_gradient_peak_size = static_weight_size

    # --------------------
    # Activation usage (original / unoptimized scenario)
    # --------------------
    # Note: get_activation_usage returns usage for 1 token, so we multiply by batch * seq_length
    single_token_activation = get_activation_usage(model)
    dynamic_activation_size = batch * seq_length * single_token_activation

    # --------------------
    # Save original GPU memory usage figure
    # --------------------
    save_memory_usage_figure_gpu(
        "original_gpu_8B_VRAM.png",
        static_weight_size / GiB,
        static_optimizer_size / GiB,
        dynamic_gradient_peak_size / GiB,
        dynamic_activation_size / GiB,
    )

    # --------------------
    # Optimized GPU memory usage
    # --------------------
    dynamic_optimized_activation_size = model_precision_size * get_llama3_8B_optimized_activation_size(batch, seq_length)
    dynamic_checkpointed_size = model_precision_size * get_llama3_8B_checkpointed_size(batch, seq_length)

    save_optimized_memory_usage_figure_gpu(
        "optimized_gpu_8B_VRAM.png",
        static_weight_size / GiB,
        static_optimizer_size / GiB,
        dynamic_gradient_peak_size / GiB,
        dynamic_optimized_activation_size / GiB,
        dynamic_checkpointed_size / GiB,
    )

    # --------------------
    # CPU Offloading scenario
    # --------------------
    cpu_offloading_forward_part_size = dynamic_optimized_activation_size
    cpu_offloading_backward_part_size = (
        model_precision_size
        * get_llama3_8B_cpuoffloading_backward_gradient_and_activation_size(batch, seq_length)
    )

    save_optimized_memory_usage_figure_cpu(
        "optimized_cpu_8B_VRAM.png",
        cpu_offloading_forward_part_size / GiB,
        cpu_offloading_backward_part_size / GiB,
    )

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
        "optimized_cpu_8B_DRAM.png",
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

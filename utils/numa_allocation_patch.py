import os
import subprocess
import torch
import deepspeed
from numa_allocation import zeros_numa_onnode_cpu, zeros_numa_on_nodemask_cpu

def get_numastat_output() -> str:
    """
    Retrieve NUMA statistics for the current process.

    Returns:
        str: Output from the `numastat` command for the current process.
    """
    def run_command(cmd: list[str]) -> str:
        """
        Run a shell command and return its output as a string.

        Args:
            cmd (list[str]): The command to run.

        Returns:
            str: The standard output from the command, or an error message.
        """
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return f"Command failed with exit code {result.returncode}"
        return result.stdout.strip()
    pid = os.getpid()
    return run_command(["numastat", "-c", "-p", str(pid)])

def zeros_cpu(shape: tuple[int], dtype: torch.dtype, pin_memory: bool) -> torch.Tensor:
    """
    This one doesn't need to be patched because it is already patched from the offload_grad_checkpoint.
    ngpus * batch * seq * hidden * dtype
    """
    return zeros_numa_onnode_cpu(shape=shape, dtype=dtype, pin_memory=pin_memory, priority_numa_nodes=[])

def zeros_cpu_for_checkpointed_striping_0(shape: tuple[int], dtype: torch.dtype, pin_memory: bool) -> torch.Tensor:
    """
    This one doesn't need to be patched because it is already patched from the offload_grad_checkpoint.
    ngpus * batch * seq * hidden * dtype
    """
    return zeros_numa_onnode_cpu(shape=shape, dtype=dtype, pin_memory=pin_memory, priority_numa_nodes=[3])

def zeros_cpu_for_checkpointed_striping_1(shape: tuple[int], dtype: torch.dtype, pin_memory: bool) -> torch.Tensor:
    """
    This one doesn't need to be patched because it is already patched from the offload_grad_checkpoint.
    ngpus * batch * seq * hidden * dtype
    """
    return zeros_numa_onnode_cpu(shape=shape, dtype=dtype, pin_memory=pin_memory, priority_numa_nodes=[3])

def _zeros_cpu_for_compute_gradients_striping_0(shape: tuple[int], dtype: torch.dtype, pin_memory: bool) -> torch.Tensor:
    """
    2 * model_size (if gradient accumulation dtype is bf16)
    """
    return zeros_numa_onnode_cpu(shape=shape, dtype=dtype, pin_memory=pin_memory, priority_numa_nodes=[3])

def _zeros_cpu_for_compute_gradients_striping_1(shape: tuple[int], dtype: torch.dtype, pin_memory: bool) -> torch.Tensor:
    """
    2 * model_size (if gradient accumulation dtype is bf16)
    """
    return zeros_numa_onnode_cpu(shape=shape, dtype=dtype, pin_memory=pin_memory, priority_numa_nodes=[3])

def _zeros_cpu_for_compute_weights_striping_0(shape: tuple[int], dtype: torch.dtype, pin_memory: bool) -> torch.Tensor:
    """
    2 * model_size (if bf16/fp16 mixed precision)
    """
    return zeros_numa_onnode_cpu(shape=shape, dtype=dtype, pin_memory=pin_memory, priority_numa_nodes=[3])

def _zeros_cpu_for_compute_weights_striping_1(shape: tuple[int], dtype: torch.dtype, pin_memory: bool) -> torch.Tensor:
    """
    2 * model_size (if bf16/fp16 mixed precision)
    """
    return zeros_numa_onnode_cpu(shape=shape, dtype=dtype, pin_memory=pin_memory, priority_numa_nodes=[3])

def _zeros_cpu_for_master_gradients(shape: tuple[int], dtype: torch.dtype, pin_memory: bool) -> torch.Tensor:
    """
    4 * model_size
    """
    return zeros_numa_onnode_cpu(shape=shape, dtype=dtype, pin_memory=pin_memory, priority_numa_nodes=[0])

def _zeros_cpu_for_master_weights(shape: tuple[int], dtype: torch.dtype, pin_memory: bool) -> torch.Tensor:
    """
    4 * model_size
    """
    return zeros_numa_onnode_cpu(shape=shape, dtype=dtype, pin_memory=pin_memory, priority_numa_nodes=[0])

def _zeros_cpu_for_momentums(shape: tuple[int], dtype: torch.dtype, pin_memory: bool) -> torch.Tensor:
    """
    4 * model_size
    """
    return zeros_numa_onnode_cpu(shape=shape, dtype=dtype, pin_memory=pin_memory, priority_numa_nodes=[])

def _zeros_cpu_for_variances(shape: tuple[int], dtype: torch.dtype, pin_memory: bool) -> torch.Tensor:
    """
    4 * model_size
    """
    return zeros_numa_onnode_cpu(shape=shape, dtype=dtype, pin_memory=pin_memory, priority_numa_nodes=[])

def patch_deepspeed_cpu_tensor_allocation(rank: int):
    print("Apply numa awareness allocation deepspeed")
    if rank % 2 == 0:
        _zeros_cpu_for_compute_gradients = _zeros_cpu_for_compute_gradients_striping_0
        _zeros_cpu_for_compute_weights = _zeros_cpu_for_compute_weights_striping_0
    else:
        _zeros_cpu_for_compute_gradients = _zeros_cpu_for_compute_gradients_striping_1
        _zeros_cpu_for_compute_weights = _zeros_cpu_for_compute_weights_striping_1
    deepspeed.ops.adam.cpu_adam.zeros_cpu_for_momentums = _zeros_cpu_for_momentums
    deepspeed.ops.adam.cpu_adam.zeros_cpu_for_variances = _zeros_cpu_for_variances
    deepspeed.runtime.zero.stage3.zeros_cpu_for_master_weights = _zeros_cpu_for_master_weights
    deepspeed.runtime.zero.stage3.zeros_cpu_for_master_gradients = _zeros_cpu_for_master_gradients
    deepspeed.runtime.zero.stage3.zeros_cpu_for_compute_weights = _zeros_cpu_for_compute_weights
    deepspeed.runtime.zero.stage3.zeros_cpu_for_compute_gradients = _zeros_cpu_for_compute_gradients
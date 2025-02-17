'''
Modified Gradient Checkpointing Utility:
Based on original implementation from Unsloth Zoo: https://github.com/unslothai/unsloth-zoo/blob/main/unsloth_zoo/gradient_checkpointing.py

Modifications: Adjusted gradient checkpointing to support additional functionalities (multiple GPU).

This program is distributed under the GNU Lesser General Public License (LGPL), version 3 or later.
You are free to redistribute and/or modify this program under the terms of the LGPL as published by the Free Software Foundation.
'''

import torch
from packaging.version import Version
torch_version = torch.__version__

if Version(torch_version) < Version("2.4.0"):
    torch_amp_custom_fwd = torch.cuda.amp.custom_fwd
    torch_amp_custom_bwd = torch.cuda.amp.custom_bwd
else:
    torch_amp_custom_fwd = torch.amp.custom_fwd(device_type="cuda")
    torch_amp_custom_bwd = torch.amp.custom_bwd(device_type="cuda")

from .numa_allocation_patch import zeros_cpu, zeros_cpu_for_checkpointed_striping_0, zeros_cpu_for_checkpointed_striping_1

class PinnedBufferManager:
    def __init__(self):
        pass
        
    def setup_buffer(self, num_layers: int, batch_size: int, seq_length: int, hidden_size: int, dtype: torch.dtype, rank: int, is_numa: bool) -> None:
        buffer_shape = (batch_size, seq_length, hidden_size)
        if rank % 2 == 0:
            zeros_cpu_for_checkpointed = zeros_cpu_for_checkpointed_striping_0
        else:
            zeros_cpu_for_checkpointed = zeros_cpu_for_checkpointed_striping_1
        if is_numa == False:    # use default policy
            zeros_cpu_for_checkpointed = zeros_cpu
        self.buffers = [zeros_cpu_for_checkpointed(shape=buffer_shape, dtype=dtype, pin_memory=True) for _ in range(num_layers)]
        
    def get_buffer(self) -> torch.Tensor:
        assert len(self.buffers) > 0, "Run out of buffer"
        return self.buffers.pop()
    
    def release_buffer(self, buffer: torch.Tensor) -> None:
        self.buffers.append(buffer)

buffer_manager = PinnedBufferManager()

class Offloaded_Gradient_Checkpointer(torch.autograd.Function):
    @staticmethod
    @torch_amp_custom_fwd
    def forward(ctx, forward_function, hidden_states, *args):
        device = hidden_states.device
        saved_hidden_states = buffer_manager.get_buffer()
        saved_hidden_states.copy_(hidden_states, non_blocking=True)
        with torch.no_grad():
            output = forward_function(hidden_states, *args)
        ctx.save_for_backward(saved_hidden_states)
        ctx.forward_function = forward_function
        ctx.args = args
        ctx.device = device
        return output

    @staticmethod
    @torch_amp_custom_bwd
    def backward(ctx, dY):
        (saved_hidden_states,) = ctx.saved_tensors
        hidden_states = saved_hidden_states.to(ctx.device, non_blocking = True).detach()
        buffer_manager.release_buffer(saved_hidden_states)
        hidden_states.requires_grad_(True)
        with torch.enable_grad():
            (output,) = ctx.forward_function(hidden_states, *ctx.args)
        torch.autograd.backward(output, dY)
        return (None, hidden_states.grad,) + (None,)*len(ctx.args)

@torch._disable_dynamo
def offloaded_gradient_checkpoint(function, *args, use_reentrant = None, **kwargs):
    return Offloaded_Gradient_Checkpointer.apply(function, *args)

def patch_offloaded_gradient_checkpointing():
    import torch.utils
    import transformers.modeling_utils
    torch.utils.checkpoint.checkpoint = offloaded_gradient_checkpoint
    transformers.modeling_utils.checkpoint = offloaded_gradient_checkpoint
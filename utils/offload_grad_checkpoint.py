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

class Offloaded_Gradient_Checkpointer(torch.autograd.Function):
    @staticmethod
    @torch_amp_custom_fwd
    def forward(ctx, forward_function, hidden_states, *args):
        device = hidden_states.device
        saved_hidden_states = hidden_states.to("cpu", non_blocking = True)
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
        (hidden_states,) = ctx.saved_tensors
        hidden_states = hidden_states.to(ctx.device, non_blocking = True).detach()
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
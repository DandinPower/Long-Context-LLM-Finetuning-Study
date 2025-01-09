import torch
from packaging.version import Version
torch_version = torch.__version__

if Version(torch_version) < Version("2.4.0"):
    torch_amp_custom_fwd = torch.cuda.amp.custom_fwd
    torch_amp_custom_bwd = torch.cuda.amp.custom_bwd
else:
    torch_amp_custom_fwd = torch.amp.custom_fwd(device_type="cuda")
    torch_amp_custom_bwd = torch.amp.custom_bwd(device_type="cuda")

class Unsloth_Offloaded_Gradient_Checkpointer(torch.autograd.Function):
    """
    This class is inspired by or based on the `Unsloth_Offloaded_Gradient_Checkpointer` implementation.

    Original code licensed under LGPL.
    Description of the original code: Saves VRAM by smartly offloading to RAM with minimal performance impact.

    Modifications:
    - Add support to multiple cuda device -> replace hard coded "cuda:0"
    - Updated documentation for clarity and adjusted logic for specific performance needs.

    Original Source: https://github.com/unslothai/unsloth-zoo/blob/main/unsloth_zoo/gradient_checkpointing.py#L145

    License: LGPL
    """
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
    pass

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
    pass
pass

@torch._disable_dynamo
def unsloth_offloaded_gradient_checkpoint(function, *args, use_reentrant = None, **kwargs):
    return Unsloth_Offloaded_Gradient_Checkpointer.apply(function, *args)
pass

def patch_unsloth_gradient_checkpointing():
    print("Patched gradient checkpointing func to offloaded gradient checkpointing!")
    import torch.utils
    import transformers.modeling_utils
    torch.utils.checkpoint.checkpoint = unsloth_offloaded_gradient_checkpoint
    transformers.modeling_utils.checkpoint = unsloth_offloaded_gradient_checkpoint
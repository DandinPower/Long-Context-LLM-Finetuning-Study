from torch.nn import Module
from transformers import AutoModelForCausalLM
from transformers.integrations.deepspeed import HfDeepSpeedConfig
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from .monkey_patch import patch_unsloth_gradient_checkpointing

def create_model_by_deepspeed(ds_config: dict, model_name: str, liger_kernel: bool, gradient_checkpointing: bool, offload_gradient_checkpointing: bool, flash_attn_2: bool) -> Module:
    assert model_name is not None, "model_name must be provided"
    assert liger_kernel is not None, "liger_kernel must be provided"
    assert gradient_checkpointing is not None, "gradient_checkpoint must be provided"
    assert flash_attn_2 is not None, "flash_attn_2 must be provided"
    
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None

    model_class = AutoModelForCausalLM
    if liger_kernel:
        model_class = AutoLigerKernelForCausalLM

    if flash_attn_2:
        model = model_class.from_pretrained(model_name, use_cache=False, attn_implementation="flash_attention_2")
    else:
        model = model_class.from_pretrained(model_name, use_cache=False)

    if offload_gradient_checkpointing:
        assert gradient_checkpointing, "Need to enable gradient_checkpointing with offload_gradient_checkpointing"
        patch_unsloth_gradient_checkpointing()

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return model
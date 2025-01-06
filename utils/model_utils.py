from torch.nn import Module
from transformers import AutoModelForCausalLM
from transformers.integrations.deepspeed import HfDeepSpeedConfig
from liger_kernel.transformers import apply_liger_kernel_to_llama

default_liger_kernel_config = {
    "rope": True,
    "swiglu": True,
    "cross_entropy": False,
    "fused_linear_cross_entropy": True,
    "rms_norm": True,
}

def create_model_by_deepspeed(ds_config: dict, model_name: str, liger_kernel: bool, gradient_checkpointing: bool, offload_gradient_checkpointing: bool, flash_attn_2: bool) -> Module:
    assert model_name is not None, "model_name must be provided"
    assert liger_kernel is not None, "liger_kernel must be provided"
    assert gradient_checkpointing is not None, "gradient_checkpoint must be provided"
    assert flash_attn_2 is not None, "flash_attn_2 must be provided"
    
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None

    if liger_kernel:
        apply_liger_kernel_to_llama(
            rope=default_liger_kernel_config["rope"],
            swiglu=default_liger_kernel_config["swiglu"],
            cross_entropy=default_liger_kernel_config["cross_entropy"],
            fused_linear_cross_entropy=default_liger_kernel_config["fused_linear_cross_entropy"],
            rms_norm=default_liger_kernel_config["rms_norm"]
        )
    
    if flash_attn_2:
        model = AutoModelForCausalLM.from_pretrained(model_name, use_cache=False, attn_implementation="flash_attention_2")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, use_cache=False)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if offload_gradient_checkpointing:
        assert gradient_checkpointing, "Need to enable gradient_checkpointing with offload_gradient_checkpointing"
        model.offload_gradient_checkpointing_enable()

    return model
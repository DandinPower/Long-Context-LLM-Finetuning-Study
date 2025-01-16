from torch.nn import Module
from transformers import AutoModelForCausalLM
from transformers.integrations.deepspeed import HfDeepSpeedConfig
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from .offload_grad_checkpoint import patch_offloaded_gradient_checkpointing
from .lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters, make_model_gradient_checkpointing_compatible

def create_model_by_deepspeed(ds_config: dict, model_name: str, lora_dim: int, liger_kernel: bool, gradient_checkpointing: bool, offload_gradient_checkpointing: bool, flash_attn_2: bool) -> Module:
    assert model_name is not None, "model_name must be provided"
    assert liger_kernel is not None, "liger_kernel must be provided"
    assert gradient_checkpointing is not None, "gradient_checkpoint must be provided"
    assert flash_attn_2 is not None, "flash_attn_2 must be provided"
    assert lora_dim is not None, "lora_dim must be provided, if not enable lora, it should be 0"
    
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

    if lora_dim > 0:
        model = convert_linear_layer_to_lora(model, "layers.", lora_dim)
        model = only_optimize_lora_parameters(model)
        model = make_model_gradient_checkpointing_compatible(model)

    if offload_gradient_checkpointing:
        assert gradient_checkpointing, "Need to enable gradient_checkpointing with offload_gradient_checkpointing"
        patch_offloaded_gradient_checkpointing()

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return model
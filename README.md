# **Long-Context-LLM-Finetuning-Study**

## 🌟 **Introduction**

In 2025, the agentic workflow of LLMs is expected to become a significant trend. However, foundational models like Llama3.1 8B 128k lack specific training for **Retrieval-Augmented Generation (RAG)** techniques or tool-calling within agents. Therefore, it is essential to finetune these models using **domain-specific long-context datasets**.

### Example: *LongCite Study* ([Reference](https://arxiv.org/abs/2409.02897))
- Researchers finetuned Llama3.1 8B on their **LongCite-45K** dataset.
- Achieved excellent performance on tasks involving:
  - Referring to multiple lengthy documents.
  - Generating accurate, citation-based answers.

However, long-context LLM finetuning faces challenges:
- **Memory requirements**: Large static memory for model weights, optimizer states, and activation memory scales with input context length.
- **Hardware limitations**: Full finetuning demands advanced techniques like context parallelism across GPUs (e.g., **4 nodes × 8×H100 GPUs per node**).

### 💡 **Objective**
This study explores cost-effective finetuning using **commodity-level hardware** like **8×V100 GPUs**, making it accessible without sacrificing performance.

## 🧠 **Key Focus Areas**

### **1. Finetuning Precision**
- **Mixed precision** for better performance:
  - FP16 (before Ampere GPUs, e.g., V100).
  - BF16 (for newer GPUs, e.g., A6000).

### **2. Parameters to Train**
- Full-parameter finetuning: 
    - **LoRA** and **QLoRA** are better at generalizing and retaining pretrained domain knowledge, but they lack the capacity to effectively adapt to target domains that differ significantly from the pretrained domain. This makes them less suitable for tasks requiring long-context understanding. ([Reference](https://arxiv.org/abs/2405.09673))
    - The pretrained dataset for **Llama3.1 8B**, while supporting **128K** token contexts, predominantly features training lengths under **2K tokens**. This mismatch between the pretrained domain and target domain introduces a significant **domain gap**, requiring full-parameter finetuning for effective learning on target tasks. ([Reference](https://arxiv.org/abs/2407.21783))

### **3. Hardware Configurations**
| Configuration | GPUs            | VRAM      | CPU Cores | RAM          | SSD         |
|---------------|-----------------|-----------|-----------|--------------|-------------|
| Setup 1       | 8×V100 16GB     | 128GB     | 92        | 481 GB       | 6.5 TB      |
| Setup 2       | 4×A6000 48GB    | 196GB     | 56        | 429.5 GB     | 1.1 TB      |

### **4. Optimization Techniques**
- **Liger kernel** for efficient computation. ([Reference](https://github.com/linkedin/Liger-Kernel))
- **Offloaded Gradient Checkpointing** (via modified `unsloth`) to move activation memory to system RAM. ([Reference](https://github.com/unslothai/unsloth-zoo/blob/main/unsloth_zoo/gradient_checkpointing.py#L145))
- **FlashAttention2** for efficient attention mechanisms. ([Reference](https://arxiv.org/abs/2307.08691))
- **ZeRO Offload** to store static memory in DRAM. ([Reference](https://arxiv.org/abs/2101.06840))

## 📊 **Results**

### Setup: **4×A6000 (48GB GPUs)**
| Metric                  | Value         |
|-------------------------|---------------|
| **Model**               | Llama3.1 8B  |
| **Context Length**      | 128000          |
| **Peak VRAM Memory(MiB)**    | 24128.66     |
| **Peak DRAM Memory(GiB)**    | 317.53       |
| **Throughput(token/s)**          | 1775.17 |
| **Batch Size**          | 1            |

## ⚙️ **Installation**

1. **Prerequisites**
   - Ensure **NVIDIA driver** and **CUDA compiler** are installed.

2. **Install Dependencies**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    pip install flash-attn
    ```

## 🔧 **Source Code Modifications**

### **1. Modifications to `transformers/modeling_utils.py`**
Add support for **offloaded gradient checkpointing**:

```python
from packaging.version import Version
torch_version = torch.__version__

if Version(torch_version) < Version("2.4.0"):
    torch_amp_custom_fwd = torch.cuda.amp.custom_fwd
    torch_amp_custom_bwd = torch.cuda.amp.custom_bwd
else:
    torch_amp_custom_fwd = torch.amp.custom_fwd(device_type="cuda")
    torch_amp_custom_bwd = torch.amp.custom_bwd(device_type="cuda")

class Unsloth_Offloaded_Gradient_Checkpointer(torch.autograd.Function):
    @staticmethod
    @torch_amp_custom_fwd
    def forward(ctx, forward_function, hidden_states, *args):
        device = hidden_states.device
        saved_hidden_states = hidden_states.to("cpu", non_blocking=True)
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
        hidden_states = hidden_states.to(ctx.device, non_blocking=True).detach()
        hidden_states.requires_grad_(True)
        with torch.enable_grad():
            output = ctx.forward_function(hidden_states, *ctx.args)
        torch.autograd.backward(output, dY)
        return (None, hidden_states.grad,) + (None,) * len(ctx.args)
```

### **2. Modifications to `src/transformers/models/llama/modeling_llama.py`**

#### Import Updates
```python
from ...modeling_utils import PreTrainedModel, Unsloth_Offloaded_Gradient_Checkpointer
```

#### Add New Attribute in `LlamaModel`
```python
class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        ...
        self.offload_gradient_checkpointing = False  # Add this
```

#### Add Enable Function
```python
def offload_gradient_checkpointing_enable(self):
    self.offload_gradient_checkpointing = True
```

#### Update Decoder Inference Logic
```python
for decoder_layer in self.layers[:self.config.num_hidden_layers]:
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    if self.gradient_checkpointing and self.training:
        if self.offload_gradient_checkpointing:
            layer_outputs = Unsloth_Offloaded_Gradient_Checkpointer.apply(
                decoder_layer,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
            )
        else:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
            )
    else:
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **flash_attn_kwargs,
        )
    hidden_states = layer_outputs[0]
```
# Long-Context-LLM-Finetuning-Study

## Introduction

In 2025, the agentic workflow of LLMs is expected to become a significant trend. However, since RAG (Retrieval-Augmented Generation) techniques or tool calling within each agent are not specifically trained by general foundational models, it is crucial to finetune foundational models, such as Llama3.1 8B 128k, using domain-specific long-context datasets.

For example, in a study like *LongCite*, researchers finetuned Llama3.1 8B on their long-context dataset, LongCite-45K, to achieve excellent performance in tasks that require referring to multiple lengthy documents and generating answers with accurate citations to the source material. However, training long-context LLMs demands substantial memory resources due to the large static memory footprint of model weights and optimizer states, as well as the significant activation memory requirements, which scale with the length of the input context. Performing full finetuning on GPUs is challenging, as it requires advanced techniques like context parallelism to distribute activation contexts across GPUs. Such an approach typically necessitates a massive hardware setup—for example, 4 nodes with 8×H100 GPUs per node.

To make finetuning more cost-effective, this study focuses on using affordable, commodity-level hardware configurations such as 8×V100 GPUs.

## Key Focus Areas of This Study

This study investigates efficient finetuning of long-context LLMs under the following configurations and techniques:

1. **Finetuning Precision**: Mixed precision (FP16/BF16)  

2. **Parameters to Train**:  
   - Full parameter finetuning, because LoRA (Low-Rank Adaptation) and QLoRA are generally not well-suited for tasks like these. Research has shown that their ability to transfer knowledge from pretrained domain models to target domains is limited, especially in scenarios requiring the processing of long contexts. Even though Llama3.1 8B is pretrained on 128K data, the majority of training lengths in the pretrained data are under 2K tokens, which means the pretrained domain is quite different from target domains.

3. **Hardware Configurations**:  
   - 8×V100 16GB GPUs (Total VRAM: 128GB)  
   - 4×A6000 48GB GPUs (Total VRAM: 196GB)  

4. **Optimization Techniques**:  
   - Liger kernel for optimized computation  
   - Offloaded gradient checkpointing  
   - FlashAttention2 for efficient attention mechanisms  
   - ZeRO offload to move static memory into DRAM  

This approach aims to reduce training costs while maintaining the effectiveness of long-context LLM finetuning.

## Installation

1. ensure you have nvidia driver and cuda compiler installed.
2. install dependencies
    `python3 -m venv venv`
    `pip3 install -r requirements.txt`
    `pip3 install flash-attn`

## Source Code Mofification

### transformers

1. For offload Gradient Checkpointing utils: `transformers/modeling_utils.py`: 
    ```python
    from packaging.version import Version
    torch_version = torch.__version__
    if Version(torch_version) < Version("2.4.0"):
        torch_amp_custom_fwd = torch.cuda.amp.custom_fwd
        torch_amp_custom_bwd = torch.cuda.amp.custom_bwd
    else:
        torch_amp_custom_fwd = torch.amp.custom_fwd(device_type = "cuda")
        torch_amp_custom_bwd = torch.amp.custom_bwd(device_type = "cuda")
    pass


    class Unsloth_Offloaded_Gradient_Checkpointer(torch.autograd.Function):
        """
        Code licensed under LGPL
        Saves VRAM by smartly offloading to RAM.
        Tiny hit to performance, since we mask the movement via non blocking calls.
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
            device = ctx.device
            hidden_states = hidden_states.to(device, non_blocking = True).detach()
            hidden_states.requires_grad_(True)
            with torch.enable_grad():
                (output,) = ctx.forward_function(hidden_states, *ctx.args)
            torch.autograd.backward(output, dY)
            return (None, hidden_states.grad,) + (None,)*len(ctx.args)
        pass
    pass
    ```

2. For the Llama Model: `src/transformers/models/llama/modeling_llama.py`

    - import necessary things
        original: `from ...modeling_utils import PreTrainedModel` into `from ...modeling_utils import PreTrainedModel, Unsloth_Offloaded_Gradient_Checkpointer`

    - in the __init__ function of LlamaModel:
        ```python
        class LlamaModel(LlamaPreTrainedModel):
            ...
            def __init__(self, config: LlamaConfig):
                ...
                self.offload_gradient_checkpointing = False # Add this line
        ```
    
    - add new member function for LlamaModel
        ```python
        def offload_gradient_checkpointing_enable(self):
            self.offload_gradient_checkpointing = True
        ```

    - add new member function for LlamaForCausalLM
        ```python
        def offload_gradient_checkpointing_enable(self):
            self.model.offload_gradient_checkpointing_enable()
        ```

    - The decoder inference part:
        ```python
        for decoder_layer in self.layers:
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
                )
        ```
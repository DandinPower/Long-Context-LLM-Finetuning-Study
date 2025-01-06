# Long-Context-LLM-Finetuning-Study

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
   - 4×A6000 48GB GPUs (Total VRAM: 96GB)  

4. **Optimization Techniques**:  
   - Liger kernel for optimized computation  
   - Offloaded gradient checkpointing  
   - FlashAttention2 for efficient attention mechanisms  
   - ZeRO offload to move static memory into DRAM  

This approach aims to reduce training costs while maintaining the effectiveness of long-context LLM finetuning.

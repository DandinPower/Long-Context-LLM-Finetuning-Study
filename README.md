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
- **FlashAttention2** FlashAttention2 provides efficient attention mechanisms, but it is only supported on GPUs from the Ampere generation onward (e.g., V100 GPUs are not supported). ([Reference](https://arxiv.org/abs/2307.08691))
- **ZeRO Offload** to store static memory in DRAM. ([Reference](https://arxiv.org/abs/2101.06840))

## 📊 **Results**

| Setup                 | Model         | Context Length | Peak VRAM Memory (MiB) | Peak DRAM Memory (GiB) | Throughput (token/s) | Batch Size |
|-----------------------|---------------|----------------|-------------------------|-------------------------|-----------------------|------------|
| **4×A6000** | Llama3.1 8B   | 128000         | 24128.66               | 317.53                 | 1775.17              | 1          |
| **8×V100**  | Llama3.1 8B   | 32768            | 7711.63                    | 286.43   | 2793.22                  | 1        |
| **8×V100**  | Llama3.1 8B   | 49152            | 10794.13                    | 351.68   | 2342.40                  | 1        |

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

## 🚀 **Run the Experiments**

### **1. Configure Settings**
- Most configuration settings are defined in `run.sh`. Additional configurations for DeepSpeed and ZeRO are found in `configs/cpu.json`.
- Key configurations in `run.sh`:
  1. **`MODEL_NAME`**: Specify the model name available on the Hugging Face Hub to use as the base model.
  2. **`NUM_GPUS`**: Adjust to the number of GPUs available to enable distributed training using DeepSpeed ZeRO and ZeRO-Offload.
  3. **`SYSTEM_TYPE`**: Used for snapshot output naming.
  4. **`PER_DEVICE_TRAIN_BATCH_SIZE`**: Sets the micro-batch size for each device; the total batch size is `PER_DEVICE_TRAIN_BATCH_SIZE × NUM_GPUS`.
  5. **`GRADIENT_ACCUMULATION_STEPS`**: Number of forward/backward passes to accumulate gradients before updating weights.
  6. **`MAX_SEQ_LENGTH`**: The desired context length for training.
  7. **Optimization Parameters**:
      - `LEARNING_RATE=1e-4`
      - `WEIGHT_DECAY=0.01`
      - `BETA_0=0.9`
      - `BETA_1=0.95`
  8. **`NUM_TRAIN_ITERATION`**: Number of iterations for the experiment. To ensure correct statistical results, set this to a value greater than 2, as the first iteration is discarded as a warm-up.
  9. Enable advanced optimization techniques:
      - `--liger_kernel` for Liger Kernel.
      - `--gradient_checkpointing` for on-GPU gradient checkpointing.
      - `offload_gradient_checkpointing` to offload checkpointed values to the CPU.
      - `--flash_attn_2` for Flash Attention 2 (Ampere GPUs and newer only).

### **2. DeepSpeed Configuration**
- Follow the default CPU offload settings. Adjust as needed for your hardware:
    ```json
    "fp16": {
        "enabled": false,
        "loss_scale_window": 100,
        "initial_scale_power": 6,
        "hysteresis": 1
    },
    "bf16": {
        "enabled": true
    }
    ```
  - Use **BF16** for GPUs from the Ampere generation or newer (e.g., A6000), and **FP16** for older GPUs (e.g., V100).
- Adjust `offload_parameter` as needed for memory management across multiple GPUs.

### **3. Run the Experiment**
1. Execute the script:
    ```bash
    bash run.sh
    ```
   Example output:
    ```log
    [RESULT] Peak VRAM Usage(per gpu): 4664.49 MB
    [RESULT] Avg Iteration Latency(total): 9.81 s
    [RESULT] Each Iteration Latency (rank0): [9.80996334599331]
    [RESULT] Tokens(total): 32768
    [RESULT] Throughput(total): 3340.28 (token/s)
    ```

2. To monitor CPU memory usage:
    ```bash
    bash memory_monitor.sh
    ```
   Run it concurrently with `run.sh` to track CPU usage.

### **4. Notes**
- Ensure compatibility with your hardware when enabling advanced features like Flash Attention 2 or BF16.
- For optimal results, experiment with different batch sizes and gradient accumulation settings.

# Long-Context LLM Finetuning Study (NUMA-Aware Version)

This repository contains benchmark code for testing enhancements to DeepSpeed ZeRO-Offload ([Link](https://github.com/deepspeedai/DeepSpeed)). The modified DeepSpeed version used here is available at ([Link](https://github.com/DandinPower/DeepSpeed-0.16.0/tree/numa_aware)). Additionally, we’ve implemented several C++ PyTorch extensions to support these enhancements.

## Installation

### 1. Prerequisites
- Ensure an **NVIDIA driver** and **CUDA compiler** are installed.
- Install the following system dependencies (via `apt`):
  ```bash
  sudo apt-get install python-dev build-essential python3-venv
  ```

### 2. Install Project Python Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install flash-attn
```

### 3. Install Enhanced DeepSpeed and C++ Kernels
- Follow the installation steps for:
  - Modified DeepSpeed: ([Link](https://github.com/DandinPower/DeepSpeed-0.16.0/tree/numa_aware))
  - Zero-Overhead Pinned Memory: ([Link](https://github.com/DandinPower/zero-overhead-pinned-memory))
  - NUMA Allocation: ([Link](https://github.com/DandinPower/numa-allocation))

### 4. (Optional) Weighted Interleave with `numactl`
1. **Kernel Requirement**: Weighted interleave support requires Linux kernel ≥ 6.9 (tested on 6.12.18). This enables settings like `/sys/kernel/mm/mempolicy/weighted_interleave/node*`. Update your kernel if using this feature.
2. **Latest `numactl`**: Weighted interleave requires `numactl` ≥ 2.0.19 (APT provides 2.0.18), so compile it manually:
   ```bash
   sudo apt remove numactl libnuma-dev
   git clone https://github.com/numactl/numactl.git
   cd numactl
   git checkout v2.0.19
   ./autogen.sh
   ./configure
   make
   make test
   sudo make install
   ```

## Configuration Settings

- **Primary Configuration**: Most settings are in `run.sh`. DeepSpeed and ZeRO-specific configurations are in `configs/*.json`. For details, see:
  - ([Link](https://github.com/DandinPower/DeepSpeed-0.16.0/tree/mem-efficient))
  - DeepSpeed Documentation ([Link](https://www.deepspeed.ai/docs/config-json/))

- **Key Settings in `run.sh`**:
  1. **`MODEL_NAME`**: Base model name from Hugging Face Hub.
  2. **`NUM_LAYERS`**: Number of decoder layers (for pre-allocating CPU offload checkpoint buffers).
  3. **`HIDDEN_SIZE`**: Model hidden size (for pre-allocating CPU offload checkpoint buffers).
  4. **`NUM_GPUS`**: Number of GPUs for distributed training with DeepSpeed ZeRO/ZeRO-Offload.
  5. **`SYSTEM_TYPE`**: Naming convention for snapshot outputs (currently disabled due to increased memory usage).
  6. **`PER_DEVICE_TRAIN_BATCH_SIZE`**: Micro-batch size per device; total batch size = `PER_DEVICE_TRAIN_BATCH_SIZE × NUM_GPUS`.
  7. **`GRADIENT_ACCUMULATION_STEPS`**: Number of forward/backward passes before weight updates.
  8. **`MAX_SEQ_LENGTH`**: Context length for training.
  9. **`LORA_DIM`**: LoRA dimension (default: 0, disabled).
  10. **Optimization Parameters**:
      - `LEARNING_RATE=1e-4`
      - `WEIGHT_DECAY=0.01`
      - `BETA_0=0.9`
      - `BETA_1=0.95`
  11. **`NUM_TRAIN_ITERATION`**: Number of experiment iterations (use >3 for reliable stats; first two are warm-up).
  12. **Advanced Optimization Flags**:
      - `--liger_kernel`: Enables Liger Kernel.
      - `--gradient_checkpointing`: Enables on-GPU gradient checkpointing.
      - `--offload_gradient_checkpointing`: Offloads checkpoints to CPU (requires `--gradient_checkpointing`).
      - `--flash_attn_2`: Enables Flash Attention 2 (Ampere GPUs or newer).
      - `--zero_overhead_pin_memory`: Enables efficient pinned memory allocation.
      - `--numa_aware_allocation`: Enables NUMA-aware allocation patching (see `utils/numa_allocation_patch.py`).

## Running the Experiment

### 1. Set NUMA Allocation Strategy (Example)

The detail numactl feature can refer to their official repo ([Link](https://github.com/numactl/numactl))

- **Global Strategy** (if `--numa_aware_allocation` is not used):
  ```bash
  echo 1 > /sys/kernel/mm/mempolicy/weighted_interleave/node0
  echo 2 > /sys/kernel/mm/mempolicy/weighted_interleave/node3
  numactl --weighted-interleave=0,3 <command>
  ```
  Or use standard interleave:
  ```bash
  numactl --interleave=0,3 <command>
  ```
- **Manual NUMA Patching**: Enable `--numa_aware_allocation` (details in `utils/numa_allocation_patch.py`).

### 2. Execute the Script
```bash
bash run.sh
```
**Example Output**:
```log
[RESULT] Peak VRAM Usage (per GPU): 4664.49 MB
[RESULT] Avg Iteration Latency (total): 9.81 s
[RESULT] Each Iteration Latency (rank0): [9.80996334599331]
[RESULT] Tokens (total): 32768
[RESULT] Throughput (total): 3340.28 (token/s)
Per-node process memory usage (in MBs) for PID 22460 (python3)
        Node 0 Node 1 Node 2 Node 3  Total
        ------ ------ ------ ------ ------
Huge         0      0      0      0      0
Heap       239      0      0    477    716
Stack        0      0      0      0      0
Private  67812      6      4 135455 203277
-------  ------ ------ ------ ------ ------
Total    68051      6      4 135932 203993
```

### 3. Monitor CPU Memory Usage
Run this concurrently with `run.sh`:
```bash
bash scripts/memory_monitor.sh
```
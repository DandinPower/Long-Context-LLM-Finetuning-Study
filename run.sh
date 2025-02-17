
# MODEL_NAME=HuggingFaceTB/SmolLM2-360M-Instruct
# MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct
# MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
# MODEL_NAME=meta-llama/Llama-3.2-1B-Instruct
# MODEL_NAME=meta-llama/Llama-3.2-3B-Instruct
# MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
# MODEL_NAME=Qwen/Qwen2.5-32B-Instruct
# MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
# MODEL_NAME=Qwen/Qwen2.5-14B-Instruct-1M
# MODEL_NAME=google/gemma-2-9b-it
# MODEL_NAME=mistralai/Mistral-Nemo-Instruct-2407
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct-1M
NUM_LAYERS=28
HIDDEN_SIZE=3584

NUM_GPUS=1
PER_DEVICE_TRAIN_BATCH_SIZE=80
GRADIENT_ACCUMULATION_STEPS=1
MAX_SEQ_LENGTH=4096

SYSTEM_TYPE=cpu_gpus1_7B
DS_CONFIG_PATH=configs/cpu.json
NUM_TRAIN_ITERATIONS=5

LORA_DIM=0

LEARNING_RATE=1e-4
WEIGHT_DECAY=0.01
BETA_0=0.9
BETA_1=0.95

# ensure the cache is clean
rm -rf ~/.cache/torch_extensions/

numactl --interleave=0,3,4 deepspeed --num_gpus $NUM_GPUS training.py --model_name $MODEL_NAME --world_size $NUM_GPUS --system_type $SYSTEM_TYPE --ds_config_path $DS_CONFIG_PATH \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --num_layers $NUM_LAYERS --hidden_size $HIDDEN_SIZE \
    --num_train_iterations $NUM_TRAIN_ITERATIONS --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS --max_seq_len $MAX_SEQ_LENGTH \
    --lora_dim $LORA_DIM \
    --learning_rate $LEARNING_RATE --weight_decay $WEIGHT_DECAY --beta_0 $BETA_0 --beta_1 $BETA_1 \
    --gradient_checkpointing \
    --offload_gradient_checkpointing \
    --liger_kernel \
    --flash_attn_2 \
    --zero_overhead_pin_memory \
    # --numa_aware_allocation
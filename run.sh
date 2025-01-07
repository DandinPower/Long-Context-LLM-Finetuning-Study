NUM_GPUS=1

# MODEL_NAME=HuggingFaceTB/SmolLM2-360M-Instruct
MODEL_NAME=meta-llama/Llama-3.2-1B-Instruct
# MODEL_NAME=meta-llama/Llama-3.2-3B-Instruct
# MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct

SYSTEM_TYPE=cpu_gpus8_1B
DS_CONFIG_PATH=configs/cpu.json

PER_DEVICE_TRAIN_BATCH_SIZE=1
NUM_TRAIN_ITERATIONS=2
GRADIENT_ACCUMULATION_STEPS=1
MAX_SEQ_LENGTH=196000

LEARNING_RATE=1e-4
WEIGHT_DECAY=0.01
BETA_0=0.9
BETA_1=0.95

# ensure the cache is clean
rm -rf ~/.cache/torch_extensions/

deepspeed --num_gpus $NUM_GPUS training.py --model_name $MODEL_NAME --world_size $NUM_GPUS --system_type $SYSTEM_TYPE --ds_config_path $DS_CONFIG_PATH \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --num_train_iterations $NUM_TRAIN_ITERATIONS --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS --max_seq_len $MAX_SEQ_LENGTH \
    --learning_rate $LEARNING_RATE --weight_decay $WEIGHT_DECAY --beta_0 $BETA_0 --beta_1 $BETA_1 \
    --liger_kernel \
    --gradient_checkpointing \
    --offload_gradient_checkpointing \
    --flash_attn_2 \
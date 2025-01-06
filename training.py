import os
import torch
import time
import deepspeed

from argparse import ArgumentParser, Namespace
from torch import distributed
from transformers import SchedulerType
from pickle import dump
from deepspeed import comm as dist

from utils.model_utils import create_model_by_deepspeed
from utils.optimizer_utils import create_optimizer
from utils.ds_utils import get_ds_config_from_path
from utils.utils import set_random_seed, print_rank_0, print_verbose, get_vocab_size, get_dummy_inputs_and_labels, get_snap_shot_name, is_offload_optimizer

SNAP_SHOT_DIRS = "snap_shots"

class DeepSpeedTrainer:
    def __init__(self):
        self.device = None
        self.tokenizer = None 
        self.model = None
        self.optimizer = None
        self.vocab_size = None

    def _set_device(self, local_rank: int) -> None:
        assert local_rank is not None, "local_rank must be provided"
        torch.cuda.set_device(local_rank)
        self.device = torch.device("cuda", local_rank)

    def _init_distributed(self, local_rank: int, world_size: int) -> None:
        assert local_rank is not None, "local_rank must be provided"
        os.environ['MASTER_ADDR'] = 'localhost'  # Use master node IP if multi-node
        os.environ['MASTER_PORT'] = '29526'     # Ensure this port is free
        distributed.init_process_group("nccl", rank=local_rank, world_size=world_size, device_id=self.device)
        deepspeed.init_distributed("nccl")

    def _add_args(self, args: Namespace) -> Namespace:
        args.global_rank = distributed.get_rank()
        args.train_batch_size = args.per_device_train_batch_size * distributed.get_world_size() * args.gradient_accumulation_steps
        args.train_micro_batch_size_per_gpu = args.per_device_train_batch_size
        return args
    
    def init(self, args: Namespace, verbose: bool=True) -> None:
        # os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = "1"
        # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.memory._record_memory_history()
        
        self._set_device(args.local_rank)
        self._init_distributed(args.local_rank, args.world_size)
        assert self.device is not None, "device must be set"
        assert args is not None, "args must be provided"

        set_random_seed(args.seed)
        args = self._add_args(args)
        ds_config = get_ds_config_from_path(args)

        self.offload_optimizer = is_offload_optimizer(ds_config)

        self.vocab_size = get_vocab_size(args.model_name)

        print_verbose(f'[DEBUG] rank[{args.local_rank}]before barrier', verbose)
        distributed.barrier()
        print_verbose(f'[DEBUG] rank[{args.local_rank}]after barrier', verbose)

        print_verbose('[INIT] Create Model', verbose)
        self.model = create_model_by_deepspeed(ds_config, model_name=args.model_name, liger_kernel=args.liger_kernel, gradient_checkpointing=args.gradient_checkpointing, \
                                               offload_gradient_checkpointing= args.offload_gradient_checkpointing, flash_attn_2=args.flash_attn_2)
        print_verbose('[INIT] Model created successfully', verbose)

        print_verbose('[INIT] Create Optimizer', verbose)
        self.optimizer = create_optimizer(self.model, lr=args.learning_rate, weight_decay=args.weight_decay, betas_0=args.beta_0, betas_1=args.beta_1, offload_optimizer=self.offload_optimizer) 
        print_verbose('[INIT] Optimizer created successfully', verbose)
        print_verbose('[INIT] DeepSpeed Engine Initialize', verbose)
        self.model, self.optimizer, _, _ = deepspeed.initialize(model=self.model, optimizer=self.optimizer, config=ds_config, dist_init_required=True)
        self.model: deepspeed.DeepSpeedEngine
        print_verbose('[INIT] DeepSpeed Engine Initialized successfully', verbose)

    def train(self, args: Namespace, verbose: bool=True) -> None:
        assert self.device is not None, "device must be set"
        assert self.model is not None, "model must be set"
        assert self.optimizer is not None, "optimizer must be set"
        assert self.vocab_size is not None, "vocab_size must be set"
        assert args is not None, "args must be provided"

        print_verbose("[TRAIN] Start Running training", verbose)

        iteration_latency = []
        self.model.train()
        for step in range(args.num_train_iterations):
            print_rank_0(f"[TRAIN] Start Running Step: {step}", args.global_rank)
            inputs, labels = get_dummy_inputs_and_labels(args.train_micro_batch_size_per_gpu, args.max_seq_len, self.vocab_size)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            start_time = time.perf_counter()
            print_rank_0(f"[TRAIN] Forward", args.global_rank)
            outputs = self.model(inputs, labels=labels)
            loss = outputs.loss
            print_rank_0(f"[TRAIN] Backward ", args.global_rank)
            self.model.backward(loss)
            print_rank_0(f"[TRAIN] Step ", args.global_rank)
            self.model.step()
            dist.barrier()
            end_time = time.perf_counter()
            iteration_latency.append(end_time - start_time)
        
        iteration_latency.pop(0)
        total_latency = sum(iteration_latency)
        max_memory = torch.cuda.max_memory_allocated()
        avg_iteration_latency = (total_latency / len(iteration_latency)) * args.gradient_accumulation_steps
        total_tokens = args.train_batch_size * args.max_seq_len
        throughput = total_tokens / avg_iteration_latency
        print(f"[RESULT] Peak VRAM Usage(per gpu): {max_memory / 1024**2:.2f} MB")
        print(f"[RESULT] Avg Iteration Latency(total): {avg_iteration_latency:.2f} s")
        print(f"[RESULT] Each Iteration Latency: {iteration_latency}")
        print(f"[RESULT] Tokens(total): {total_tokens}")
        print(f"[RESULT] Throughput(total): {throughput:.2f} (token/s)")

        snapshot = torch.cuda.memory._snapshot()
        with open(f"./{SNAP_SHOT_DIRS}/{get_snap_shot_name(args)}.pickle", 'wb') as f:
            dump(snapshot, f)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank of the process in distributed training.', required=True)
    parser.add_argument('--world_size', type=int, default=-1, required=True)
    parser.add_argument('--model_name', type=str, help='Model name or path to load from.', required=True)
    parser.add_argument('--system_type', type=str, required=True)
    parser.add_argument('--ds_config_path', type=str, help='DeepSpeed Config path to load from.', required=True)
    parser.add_argument("--per_device_train_batch_size", type=int, help="Batch size (per device) for the fake training iteration.", required=True)
    parser.add_argument("--num_train_iterations", type=int, help="The number of iterations to train for.", required=True)
    parser.add_argument('--gradient_accumulation_steps', type=int, help='Gradient accumulation steps.', required=True)
    parser.add_argument("--max_seq_len", type=int, help="The maximum sequence length.", required=True)
    parser.add_argument('--learning_rate', type=float, help='Learning rate.', required=True)
    parser.add_argument('--weight_decay', type=float, help='Weight decay.', required=True)
    parser.add_argument('--beta_0', type=float, help='Beta 0.', required=True)
    parser.add_argument('--beta_1', type=float, help='Beta 1.', required=True)
    parser.add_argument("--liger_kernel", action='store_true', help='Enable Liger Kernel for Llama model.')
    parser.add_argument('--gradient_checkpointing', action='store_true', help='Enable HF gradient checkpointing for model.')
    parser.add_argument('--offload_gradient_checkpointing', action='store_true', help='Enable Unsloth offload gradient checkpointing for model.')
    parser.add_argument('--flash_attn_2', action='store_true', help='Enable Flash Attention 2.')
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="cosine", help="The scheduler type to use.", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    
    os.makedirs(SNAP_SHOT_DIRS, exist_ok=True)

    trainer = DeepSpeedTrainer()
    trainer.init(args)
    trainer.train(args)
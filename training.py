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
from utils.numa_allocation_patch import patch_deepspeed_cpu_tensor_allocation, get_numastat_output
from utils import offload_grad_checkpoint

from zero_overhead_pinned_memory import patch_deepspeed_zero_overhead_pinned_memory
from tqdm import tqdm

SNAP_SHOT_DIRS = "snap_shots"

class DeepSpeedTrainer:
    def __init__(self):
        self.device = None
        self.tokenizer = None 
        self.model = None
        self.optimizer = None
        self.vocab_size = None
        self.fwd_progress_bar = None
        self.bwd_progress_bar = None

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
    
    def _init_progress_bar(self, global_rank: int) -> None:
        # Hook functions
        def _on_layer_forward(module, inp, out, layer_idx):
            self.fwd_progress_bar.update(1)
            self.fwd_progress_bar.set_description(f"[TRAIN] Forward – Layer {layer_idx+1}/{self.num_layers}")
            return None

        def _on_layer_backward(module, grad_input, grad_output, layer_idx):
            self.bwd_progress_bar.update(1)
            self.bwd_progress_bar.set_description(f"[TRAIN] Backward – Layer {layer_idx+1}/{self.num_layers}")
            return None
        
        def _fake_layer_forward(module, inp, out, layer_idx):
            return None

        def _fake_layer_backward(module, grad_input, grad_output, layer_idx):
            return None
        
        # if global_rank == 0:    # only log in global_rank 0
        self.num_layers = len(self.model.model.layers)
        for idx, layer in enumerate(self.model.model.layers):
            if global_rank == 0:
                fh = layer.register_forward_hook(lambda mod, inp, out, idx=idx: _on_layer_forward(mod, inp, out, idx))
                bh = layer.register_full_backward_hook(lambda mod, gin, gout, idx=idx: _on_layer_backward(mod, gin, gout, idx))
            else:   # to prevent the wrong synchronize behavior
                fh = layer.register_forward_hook(lambda mod, inp, out, idx=idx: _fake_layer_forward(mod, inp, out, idx))
                bh = layer.register_full_backward_hook(lambda mod, gin, gout, idx=idx: _fake_layer_backward(mod, gin, gout, idx))
    
    def _start_progress_bar(self, is_fwd: bool, global_rank: int) -> None:
        if global_rank == 0:
            if is_fwd:
                self.fwd_progress_bar = tqdm(total=self.num_layers, desc="[TRAIN] Forward Decoder Layer Progress")  # total steps = embedding + layers + lm_head
            else:
                self.bwd_progress_bar = tqdm(total=self.num_layers, desc="[TRAIN] Backward Decoder Layer Progress")  # total steps = embedding + layers + lm_head

    def _stop_progress_bar(self, is_fwd: bool, global_rank: int) -> None:
        if global_rank == 0:
            if is_fwd:
                self.fwd_progress_bar.close()
            else:
                self.bwd_progress_bar.close()

    def init(self, args: Namespace, verbose: bool=True) -> None:
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

        offload_grad_checkpoint.buffer_manager.setup_buffer(num_layers=args.num_layers, batch_size=args.train_micro_batch_size_per_gpu, seq_length=args.max_seq_len, hidden_size=args.hidden_size, dtype=torch.bfloat16, rank=args.global_rank, is_numa=args.numa_aware_allocation)

        if args.zero_overhead_pin_memory:
            patch_deepspeed_zero_overhead_pinned_memory()

        if args.numa_aware_allocation:
            patch_deepspeed_cpu_tensor_allocation(args.global_rank)

        print_verbose('[INIT] Create Model', verbose)
        if args.lora_dim > 0:
            print_verbose(f'[INIT] Create LoRA Model with dim: {args.lora_dim}', verbose)
        self.model = create_model_by_deepspeed(ds_config, model_name=args.model_name, lora_dim=args.lora_dim, liger_kernel=args.liger_kernel, gradient_checkpointing=args.gradient_checkpointing, \
                                               offload_gradient_checkpointing= args.offload_gradient_checkpointing, flash_attn_2=args.flash_attn_2)
        print_verbose('[INIT] Model created successfully', verbose)
        
        self._init_progress_bar(args.global_rank)

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

        print_verbose(f"[TRAIN] Start Running training on pid: {os.getpid()}", verbose)

        iteration_latency = []
        fwd_duration = []
        bwd_duration = []
        step_duration = []
        self.model.train()
        for step in range(args.num_train_iterations):
            print_rank_0(f"[TRAIN] Start Running Step: {step}", args.global_rank)
            inputs, labels = get_dummy_inputs_and_labels(args.train_micro_batch_size_per_gpu, args.max_seq_len, self.vocab_size)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            start_time = time.perf_counter()
            print_rank_0(f"[TRAIN] Forward Start", args.global_rank)
            self._start_progress_bar(is_fwd=True, global_rank=args.global_rank)
            temp_start_time = time.perf_counter()
            ### FWD
            outputs = self.model(inputs, labels=labels)
            loss = outputs.loss
            ### FWD
            temp_end_time = time.perf_counter()
            temp_duration = temp_end_time - temp_start_time
            self._stop_progress_bar(is_fwd=True, global_rank=args.global_rank)
            fwd_duration.append(temp_duration)
            print_rank_0(f"[TRAIN] Forward Stop - Duration {temp_duration:.2f}s", args.global_rank)
            print_rank_0(f"[TRAIN] Backward Start", args.global_rank)
            self._start_progress_bar(is_fwd=False, global_rank=args.global_rank)
            temp_start_time = time.perf_counter()
            ### BWD
            self.model.backward(loss)
            ### BWD
            temp_end_time = time.perf_counter()
            temp_duration = temp_end_time - temp_start_time
            self._stop_progress_bar(is_fwd=False, global_rank=args.global_rank)
            bwd_duration.append(temp_duration)
            print_rank_0(f"[TRAIN] Backward Stop - Duration {temp_duration:.2f}s", args.global_rank)
            print_rank_0(f"[TRAIN] Optimizer Step Start", args.global_rank)
            temp_start_time = time.perf_counter()
            ### STEP
            self.model.step()
            dist.barrier()
            ### STEP
            end_time = time.perf_counter()
            temp_end_time = time.perf_counter()
            temp_duration = temp_end_time - temp_start_time
            step_duration.append(temp_duration)
            print_rank_0(f"[TRAIN] Optimizer Step Stop - Duration {temp_duration:.2f}s", args.global_rank)
            iteration_latency.append(end_time - start_time)
        
        iteration_latency.pop(0)
        fwd_duration.pop(0)
        bwd_duration.pop(0)
        step_duration.pop(0)
        total_latency = sum(iteration_latency)
        avg_fwd_duration = sum(fwd_duration) / len(fwd_duration)
        avg_bwd_duration = sum(bwd_duration) / len(bwd_duration)
        avg_step_duration = sum(step_duration) / len(step_duration)
        max_memory = torch.cuda.max_memory_allocated()
        avg_iteration_latency = (total_latency / len(iteration_latency)) * args.gradient_accumulation_steps
        total_tokens = args.train_batch_size * args.max_seq_len
        throughput = total_tokens / avg_iteration_latency
        print_rank_0(f"[RESULT] Peak VRAM Usage(per gpu): {max_memory / 1024**2:.2f} MB", args.global_rank)
        print_rank_0(f"[RESULT] Avg Iteration Latency(total): {avg_iteration_latency:.2f} s", args.global_rank)
        print_rank_0(f"[RESULT] Each Iteration Latency (rank0): {iteration_latency}", args.global_rank)
        print_rank_0(f"[RESULT] Avg FWD,BWD,STEP Duration: {avg_fwd_duration:.2f},{avg_bwd_duration:.2f},{avg_step_duration:.2f} s", args.global_rank)
        print_rank_0(f"[RESULT] Tokens(total): {total_tokens}", args.global_rank)
        print_rank_0(f"[RESULT] Throughput(total): {throughput:.2f} (token/s)", args.global_rank)

        print(get_numastat_output())

        # snapshot = torch.cuda.memory._snapshot()
        # with open(f"./{SNAP_SHOT_DIRS}/{get_snap_shot_name(args)}_rank{args.global_rank}.pickle", 'wb') as f:
        #     dump(snapshot, f)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank of the process in distributed training.', required=True)
    parser.add_argument('--world_size', type=int, default=-1, required=True)
    parser.add_argument('--model_name', type=str, help='Model name or path to load from.', required=True)
    parser.add_argument("--num_layers", type=int, help="The model decoder layer.", required=True)
    parser.add_argument("--hidden_size", type=int, help="The model hidden size.", required=True)
    parser.add_argument('--system_type', type=str, required=True)
    parser.add_argument('--ds_config_path', type=str, help='DeepSpeed Config path to load from.', required=True)
    parser.add_argument("--per_device_train_batch_size", type=int, help="Batch size (per device) for the fake training iteration.", required=True)
    parser.add_argument("--num_train_iterations", type=int, help="The number of iterations to train for.", required=True)
    parser.add_argument('--gradient_accumulation_steps', type=int, help='Gradient accumulation steps.', required=True)
    parser.add_argument("--max_seq_len", type=int, help="The maximum sequence length.", required=True)
    parser.add_argument("--lora_dim", type=int, help="Specifies the LoRA dimension. A value of 0 indicates that it is disabled.", default=0)
    parser.add_argument('--learning_rate', type=float, help='Learning rate.', required=True)
    parser.add_argument('--weight_decay', type=float, help='Weight decay.', required=True)
    parser.add_argument('--beta_0', type=float, help='Beta 0.', required=True)
    parser.add_argument('--beta_1', type=float, help='Beta 1.', required=True)
    parser.add_argument("--liger_kernel", action='store_true', help='Enable Liger Kernel for Llama model.')
    parser.add_argument('--gradient_checkpointing', action='store_true', help='Enable HF gradient checkpointing for model.')
    parser.add_argument('--offload_gradient_checkpointing', action='store_true', help='Enable Unsloth offload gradient checkpointing for model.')
    parser.add_argument('--flash_attn_2', action='store_true', help='Enable Flash Attention 2.')
    parser.add_argument('--zero_overhead_pin_memory', action='store_true', help='Enable Zero Overhead Pinned Memory for deepspeed.')
    parser.add_argument('--numa_aware_allocation', action='store_true', help='Enable NUMA aware allocation for cpu tensor.')
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="cosine", help="The scheduler type to use.", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    
    os.makedirs(SNAP_SHOT_DIRS, exist_ok=True)

    trainer = DeepSpeedTrainer()
    trainer.init(args)
    trainer.train(args)
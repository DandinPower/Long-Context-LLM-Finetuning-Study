import random
import numpy as np
import torch
from transformers import set_seed, AutoTokenizer
from typing import Tuple
from argparse import Namespace

def get_vocab_size(model_name: str) -> int:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab = tokenizer.get_vocab()
    return len(vocab)

def get_dummy_inputs_and_labels(batch_size: int, max_seq_length: int, vocab_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.randint(0, vocab_size, (batch_size, max_seq_length))
    labels = torch.randint(0, vocab_size, (batch_size, max_seq_length))
    return inputs, labels

def set_random_seed(seed):
    assert seed is not None, "seed must be provided"
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def print_rank_0(msg: str, rank: int) -> None:
    assert rank is not None, "rank must be provided"
    if rank == 0:
        print(msg)

def print_verbose(msg: str, verbose: bool) -> None:
    if verbose:
        print(msg)

def is_offload_optimizer(config_dict: dict) -> bool:
    zero_config_dict = config_dict["zero_optimization"]

    if "offload_optimizer" in zero_config_dict.keys():
        return True
    return False

def get_snap_shot_name(args: Namespace) -> str:
    def get_gradient_checkpointing_type(args: Namespace) -> str:
        if args.gradient_checkpointing:
            if args.offload_gradient_checkpointing:
                return "offload"
            return "normal"
        else:
            return "None"

    return f"{args.system_type}_bs{args.train_batch_size}_seq{args.max_seq_len}_liger{args.liger_kernel}_gradcheck{get_gradient_checkpointing_type(args)}_fa2{args.flash_attn_2}"
import os
import json
from argparse import Namespace

def get_ds_config_from_path(args: Namespace) -> dict:
    assert os.path.exists(args.ds_config_path), "Please provide the correct Path for config!"
    with open(args.ds_config_path, 'r') as file:
        config_dict = json.load(file)
    config_dict["train_batch_size"] = args.train_batch_size
    config_dict["train_micro_batch_size_per_gpu"] = args.train_micro_batch_size_per_gpu
    return config_dict
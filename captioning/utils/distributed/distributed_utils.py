from pathlib import Path
import copy
import torch.distributed as dist
from typing import Dict, Tuple, List

import torch
import os

def print_r0(message: str, *args, **kwargs):
    """
    Prints a message in the rank 0 process
    :param message: The message to print
    """
    if dist.get_rank() == 0:
        print(message, *args, **kwargs)

def make_directory_rank_0(directory: str):
    """
    Creates a directory in the specified path in the rank 0 process.
    The directory is immediately visible to all processes after the call
    :param directory: The path to the directory to create
    """
    if dist.get_rank() == 0:
        Path(directory).mkdir(parents=True, exist_ok=True)

    # Ensures the directory is visible to all processes after this point
    dist.barrier()

def initialize_distributed():
    """
    Initializes the distributed environment
    """
    # Initializes the distributed environment
    backend = 'nccl'
    torch.distributed.init_process_group(backend=backend)
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))

def cleanup_distributed():
    """
    Cleans up the distributed environment
    """
    torch.distributed.destroy_process_group()

def gather_value_dictionary(value_dictionary: Dict[str, float]):
    """
    Given a dictionary contining numeric values under different keys, returns a dictionary with the averaged keys over all processes
    Keys that are not present are not considered for averaging
    """

    if not torch.distributed.is_initialized():
        return copy.deepcopy(value_dictionary)

    world_size = torch.distributed.get_world_size()

    # Gathers dictionaries from all processes
    all_dictionaries = [None] * world_size
    torch.distributed.all_gather_object(all_dictionaries, value_dictionary)

    all_keys = set()
    for current_dictionary in all_dictionaries:
        all_keys.update(current_dictionary.keys())

    gathered_dictionary = {}
    for current_key in all_keys:
        gathered_dictionary[current_key] = 0.0
        counter = 0
        for current_dictionary in all_dictionaries:
            if current_key in current_dictionary:
                value = current_dictionary[current_key]
                gathered_dictionary[current_key] += value
                counter += 1
        gathered_dictionary[current_key] /= counter
    
    return gathered_dictionary

import argparse
import importlib
import os
from pathlib import Path
import sys
from typing import List, Dict, Any

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.multiprocessing
import torch.distributed as dist
import torchvision
import numpy as np
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.profilers import PyTorchProfiler
import tqdm
import time
import glob
import json
from typing import List, Dict
from evaluation.metrics.distributed_fid import DistributedFID
from evaluation.metrics.distributed_fvd import DistributedFVD
import pickle

import utils
import utils.noise_correlation
from utils.configuration.base_configuration import BaseConfiguration
from utils.configuration.evaluation_configuration import EvaluationConfiguration
from utils.distributed.distributed_utils import cleanup_distributed, initialize_distributed, make_directory_rank_0, print_r0
from utils.noise_correlation.video_region import VideoRegion
import utils.profiling
from dataset.data_modules.data_module import DataModule
from utils.checkpoint_manager import CheckpointManager
from utils.configuration.configuration import Configuration
from utils.logging.logger import Logger
from utils.tensors.tensor_folder import TensorFolder

torch.backends.cudnn.benchmark = True

def initialize_regions(video_height: int, video_width: int, video_length: int) -> List[VideoRegion]:

    all_regions = []
    current_length = 16
    while current_length <= video_length:
        current_height = 32
        current_width = 32
        while current_height <= video_height and current_width <= video_width:
            current_region = VideoRegion(current_height, current_width, current_length)
            all_regions.append(current_region)
            current_height *= 2
            current_width *= 2
        current_length *= 2
    return all_regions

def main():

    # Loads configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--variances_config", type=str, required=True)
    arguments = parser.parse_args()

    # Gets the configuration
    variances_config = arguments.variances_config
    variances_config = BaseConfiguration(variances_config).get_config()

    # Instantiates the datasets
    data_config = variances_config["data"]
    data_module = DataModule(data_config)
    data_module.setup()
    # The part of the dataset to use for sampling
    data_split = variances_config["data_split"]
    # Gets the dataloader
    dataloader = data_module.dataloader(data_split)

    video_height = variances_config["video_height"]
    video_width = variances_config["video_width"]
    video_length = variances_config["video_length"]

    output_filename = variances_config["output_filename"]
    output_directory = os.path.dirname(output_filename)
    minimum_videos = variances_config["minimum_videos"]

    # Performs initialization of the distributed environment
    initialize_distributed()

    # Makes the output directory
    print_r0("- Making output directory {}".format(output_directory))
    make_directory_rank_0(output_directory)

    regions = initialize_regions(video_height, video_width, video_length)

    # Accumulates the features from the dataloader
    last_accumulated_videos = 0
    total_accumulated_videos = 0
    print_r0("- Accumulating features")
    with tqdm.tqdm(total=minimum_videos, disable=(dist.get_rank() != 0)) as pbar:
        for current_batch in tqdm.tqdm(dataloader, disable=(dist.get_rank() != 0)):

            images = current_batch.data["video"]["video"]
            images = images.to("cuda")

            current_videos_count = images.shape[0]

            for current_region in regions:
                regions_tensor = current_region.video_to_regions(images)
                mean_differences = current_region.compute_mean_differences(regions_tensor)
                current_region.accumulate_moments(mean_differences)

            dist.barrier()
            current_videos_count = torch.as_tensor(current_videos_count, dtype=torch.int64, device="cuda")
            # Uses distirbuted to get the current images from all processes
            torch.distributed.all_reduce(current_videos_count)

            total_accumulated_videos += int(current_videos_count.item())

            # Checks if a sufficient number of frames have been accumulated
            if total_accumulated_videos >= minimum_videos:
                break

            # Updates tqdm
            pbar.update(total_accumulated_videos - last_accumulated_videos)
            last_accumulated_videos = total_accumulated_videos

    # Computes the statistics
    for current_region in regions:
        current_region.compute_statistics()
        current_region.erase_accumulated_moments()

    if dist.get_rank() == 0:
        print_r0("- Saving region statistics to {}".format(output_filename))
        # Saves the statistics
        with open(output_filename, "wb") as file:
            pickle.dump(regions, file)

    dist.barrier()
    print_r0("- Done")
    # Cleans up the distributed environment
    cleanup_distributed()

if __name__ == "__main__":
    main()

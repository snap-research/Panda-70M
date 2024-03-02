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
from utils.distributed.distributed_statistics_accumulator import DistributedStatisticsAccumulator
import utils.noise_correlation
from utils.configuration.base_configuration import BaseConfiguration
from utils.configuration.evaluation_configuration import EvaluationConfiguration
from utils.distributed.distributed_utils import cleanup_distributed, initialize_distributed, make_directory_rank_0, print_r0
from utils.noise_correlation.covariance_pair_sampler import CovariancePairSampler
from utils.noise_correlation.video_region import VideoRegion
import utils.profiling
from dataset.data_modules.data_module import DataModule
from utils.checkpoint_manager import CheckpointManager
from utils.configuration.configuration import Configuration
from utils.logging.logger import Logger
from utils.tensors.tensor_folder import TensorFolder

torch.backends.cudnn.benchmark = True

def compute_distances(video_height: int, video_width, video_length: int):
    """
    Computes the distances to use for sampling covariance pairs
    """
    names_to_dictionaries = {}
    current_distance = 2
    while current_distance <= video_height:
        names_to_dictionaries[f"h_{current_distance-1}"] = {"h_space": current_distance-1}
        current_distance *= 2
    current_distance = 2
    while current_distance <= video_width:
        names_to_dictionaries[f"w_{current_distance-1}"] = {"w_space": current_distance-1}
        current_distance *= 2
    current_distance = 2
    while current_distance <= video_length:
        names_to_dictionaries[f"t_{current_distance-1}"] = {"t_space": current_distance-1}
        current_distance *= 2

    return names_to_dictionaries

def main():

    # Loads configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--correlation_config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    arguments = parser.parse_args()
    device = arguments.device

    # Gets the configuration
    correlation_config = arguments.correlation_config
    correlation_config = BaseConfiguration(correlation_config).get_config()

    # Instantiates the datasets
    data_config = correlation_config["data"]
    data_module = DataModule(data_config)
    data_module.setup()
    # The part of the dataset to use for sampling
    data_split = correlation_config["data_split"]
    # Gets the dataloader
    dataloader = data_module.dataloader(data_split)

    video_height = correlation_config["video_height"]
    video_width = correlation_config["video_width"]
    video_length = correlation_config["video_length"]

    base_statistics_filename = correlation_config["base_statistics_output_filename"]
    output_filename = correlation_config["correlations_output_filename"]
    output_directory = os.path.dirname(output_filename)
    minimum_videos = correlation_config["minimum_videos"]

    # Performs initialization of the distributed environment
    initialize_distributed()

    # Makes the output directory
    print_r0("- Making output directory {}".format(output_directory))
    make_directory_rank_0(output_directory)

    # Reads the base statistics
    with open(base_statistics_filename, "rb") as base_statistics_file:
        base_statistics = pickle.load(base_statistics_file)
    dataset_mean = base_statistics["mean"]
    # (3) tensor with the dataset mean
    dataset_mean = torch.from_numpy(dataset_mean).to(device)
    dataset_variance = base_statistics["variance"]
    dataset_variance = torch.from_numpy(dataset_variance).to(device)

    distances_to_sample = compute_distances(video_height, video_width, video_length)
    accumulators = {
        name: DistributedStatisticsAccumulator() for name, _ in distances_to_sample.items()
    }

    # Accumulates the features from the dataloader
    last_accumulated_videos = 0
    total_accumulated_videos = 0
    print_r0("- Accumulating features")
    with tqdm.tqdm(total=minimum_videos, disable=(dist.get_rank() != 0)) as pbar:
        for current_batch in tqdm.tqdm(dataloader, disable=(dist.get_rank() != 0)):

            images = current_batch.data["video"]["video"]
            images = images.to(device)

            # Centers the images for covariance computation
            images = images - dataset_mean.unsqueeze(-1).unsqueeze(-1)

            current_videos_count = images.shape[0]

            for name, sampling_parameters in distances_to_sample.items():
                current_accumulator = accumulators[name]

                # (batch_size, samples_count, channels, 2)
                sampled_pairs = CovariancePairSampler.sample_pairs(images, **sampling_parameters)
                # The mean is already subtracted, so we just multiply to obtain covariance estimates
                samples_product = sampled_pairs[..., 0] * sampled_pairs[..., 1]
                if torch.any(torch.isnan(samples_product)):
                    print("Nans")
                if torch.any(torch.isinf(samples_product)):
                    print("Infs")

                # Accumulates the statistics
                current_accumulator.accumulate(samples_product)

            dist.barrier()
            current_videos_count = torch.as_tensor(current_videos_count, dtype=torch.int64, device=device)
            # Uses distirbuted to get the current images from all processes
            torch.distributed.all_reduce(current_videos_count)

            total_accumulated_videos += int(current_videos_count.item())

            # Checks if a sufficient number of frames have been accumulated
            if total_accumulated_videos >= minimum_videos:
                break

            # Updates tqdm
            pbar.update(total_accumulated_videos - last_accumulated_videos)
            last_accumulated_videos = total_accumulated_videos

    results = {}
    for name, accumulator in accumulators.items():
        covariance = accumulator.compute_statistics()
        results[name] = covariance.cpu().numpy()

    if dist.get_rank() == 0:
        print_r0("- Saving covariance statistics to {}".format(output_filename))
        # Saves the statistics
        with open(output_filename, "wb") as file:
            pickle.dump(results, file)

    dist.barrier()
    print_r0("- Done")
    # Cleans up the distributed environment
    cleanup_distributed()

if __name__ == "__main__":
    main()

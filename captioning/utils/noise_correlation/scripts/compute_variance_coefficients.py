import argparse
import importlib
import os
from pathlib import Path
import sys
from typing import List, Dict, Any, Tuple

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
import sklearn
import sklearn.linear_model

import utils
from utils.distributed.distributed_statistics_accumulator import DistributedStatisticsAccumulator
import utils.noise_correlation
from utils.configuration.base_configuration import BaseConfiguration
from utils.configuration.evaluation_configuration import EvaluationConfiguration
from utils.distributed.distributed_utils import cleanup_distributed, initialize_distributed, make_directory_rank_0, print_r0
from utils.noise_correlation.correlated_noise_region import CorrelatedNoiseRegion
from utils.noise_correlation.covariance_pair_sampler import CovariancePairSampler
from utils.noise_correlation.video_region import VideoRegion
import utils.profiling
from dataset.data_modules.data_module import DataModule
from utils.checkpoint_manager import CheckpointManager
from utils.configuration.configuration import Configuration
from utils.logging.logger import Logger
from utils.tensors.tensor_folder import TensorFolder

torch.backends.cudnn.benchmark = True

def covariance_name_to_distances(correlation_name: str):
    """
    Converts the correlation name to distances
    """
    result = {
        "h_dist": 0,
        "w_dist": 0,
        "t_dist": 0
    }

    if correlation_name.startswith("h"):
        result["h_dist"] = int(correlation_name[2:])
    elif correlation_name.startswith("w"):
        result["w_dist"] = int(correlation_name[2:])
    elif correlation_name.startswith("t"):
        result["t_dist"] = int(correlation_name[2:])
    else:
        raise Exception("Unknown correlation type {}".format(correlation_name))

    return result

def filter_covariances(config: Dict, covariances: Dict):
    """
    Filters away the computed covariances that should not be used for the current configuration
    """

    filtered_dict = {}

    use_temporal_regions = config["use_temporal_regions"]
    use_spatiotemporal_regions = config["use_spatiotemporal_regions"]

    for current_covariance_name, current_covariance in covariances.items():
        current_distances = covariance_name_to_distances(current_covariance_name)

        if not (use_temporal_regions or use_spatiotemporal_regions) and current_distances["t_dist"] > 0:
            continue

        filtered_dict[current_covariance_name] = current_covariance
    return filtered_dict


def make_regions(config: Dict) -> List[CorrelatedNoiseRegion]:
    """
    Builds the correlated noise regions
    """
    video_height = config["video_height"]
    video_width = config["video_width"]
    video_length = config["video_length"]

    use_temporal_regions = config["use_temporal_regions"]
    use_spatiotemporal_regions = config["use_spatiotemporal_regions"]

    all_regions = []
    current_length = 1

    while current_length <= video_length:

        current_height = 1
        current_width = 1
        while current_width <= video_width and current_height <= video_height:
            current_region = CorrelatedNoiseRegion(current_height, current_width, current_length)

            skip = False
            # Temporal or spatiotemporal needs to be set to use temporal
            if current_length > 1 and not (use_temporal_regions or use_spatiotemporal_regions):
                skip = True
            # Spatiotemporal needs to be set to use both spatial and temporal
            elif current_length > 1 and (current_width > 1 or current_height > 1) and not use_spatiotemporal_regions:
                skip = True

            if not skip:
                all_regions.append(current_region)

            current_height *= 2
            current_width *= 2
        current_length *= 2

    all_regions = [
        CorrelatedNoiseRegion(1, 1, 1),
        CorrelatedNoiseRegion(2, 2, 1),
        CorrelatedNoiseRegion(4, 4, 1),
        #CorrelatedNoiseRegion(8, 8, 1),
        #CorrelatedNoiseRegion(16, 16, 1),
        #CorrelatedNoiseRegion(32, 32, 1),
        #CorrelatedNoiseRegion(1, 1, 2),
        #CorrelatedNoiseRegion(1, 1, 4),
        #CorrelatedNoiseRegion(1, 1, 8),
        #CorrelatedNoiseRegion(1, 1, 16)
    ]

    return all_regions

def make_variance_equation(variance: np.ndarray, regions: List[CorrelatedNoiseRegion]) -> Tuple[np.ndarray, np.ndarray]:

    regions_count = len(regions)
    # The variance of all regions must sum to the variance
    return np.ones((regions_count,), dtype=np.float32), np.asarray(variance, dtype=np.float32)

def make_covariance_equation(covariance: np.ndarray, covariance_name: str, regions: List[CorrelatedNoiseRegion]) -> Tuple[np.ndarray, np.ndarray]:
    distances = covariance_name_to_distances(covariance_name)
    regions_count = len(regions)
    # The covariance of all regions that are not independent must sum to the covariance
    result = np.zeros((regions_count,), dtype=np.float32)
    for region_index, region in enumerate(regions):
        if region.is_independent_from(**distances):
            result[region_index] = 0.0
        else:
            result[region_index] = 1.0

    return result, np.asarray(covariance, dtype=np.float32)


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

    video_height = correlation_config["video_height"]
    video_width = correlation_config["video_width"]
    video_length = correlation_config["video_length"]

    base_statistics_filename = correlation_config["base_statistics_output_filename"]
    correlations_filename = correlation_config["correlations_output_filename"]
    output_filename = correlation_config["variance_coefficients_output_filename"]
    output_directory = os.path.dirname(output_filename)
    minimum_videos = correlation_config["minimum_videos"]

    Path(output_directory).mkdir(parents=True, exist_ok=True)

    # Reads the base statistics
    with open(base_statistics_filename, "rb") as base_statistics_file:
        base_statistics = pickle.load(base_statistics_file)
    dataset_mean = base_statistics["mean"]
    # (3) tensor with the dataset mean
    dataset_variance = base_statistics["variance"]


    # Reads the correlation statistics
    with open(correlations_filename, "rb") as correlations_file:
        covariances = pickle.load(correlations_file)
    # Filters the covariances that should be used for the current computation
    covariances = filter_covariances(correlation_config, covariances)

    for current_covariance_name in covariances:
        if np.any(covariances[current_covariance_name] < 0.0):
            raise ValueError("Covariance {} is negative: {}".format(current_covariance_name, covariances[current_covariance_name]))

    # Sums over channels
    dataset_variance = dataset_variance.sum()
    for current_covariance_name in covariances:
        covariances[current_covariance_name] = covariances[current_covariance_name].sum()

    covariance_keys = list(sorted(covariances.keys()))

    all_regions = make_regions(correlation_config)
    all_x_rows = []
    all_y_rows = []

    current_x, current_y = make_variance_equation(dataset_variance, all_regions)
    all_x_rows.append(current_x)
    all_y_rows.append(current_y)
    for current_covariance_key in covariance_keys:
        current_covariance = covariances[current_covariance_key]
        current_x, current_y = make_covariance_equation(current_covariance, current_covariance_key, all_regions)
        if np.any(current_x > 0.0):
            all_x_rows.append(current_x)
            all_y_rows.append(current_y)

    all_x_rows = np.stack(all_x_rows, axis=0)
    all_y_rows = np.stack(all_y_rows, axis=0)

    model = sklearn.linear_model.LinearRegression(fit_intercept=False, positive=True)
    regressed_model = model.fit(all_x_rows, all_y_rows)
    score = regressed_model.score(all_x_rows, all_y_rows)
    variance_coefficients = regressed_model.coef_

    normalized_variance_coefficients = variance_coefficients / variance_coefficients.sum()
    normalized_standard_deviations = np.sqrt(normalized_variance_coefficients)

    print("The regressed coefficients are: {}".format(normalized_variance_coefficients))
    print("The regressed standard deviations are: {}".format(normalized_standard_deviations))
    print("The regression score is {}".format(score))


if __name__ == "__main__":
    main()

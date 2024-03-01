import os
from typing import Dict, Any, Tuple

import lightning.pytorch as pl
import torch
import torch.distributed as dist
import torch.nn as nn
import numpy as np

import json
import tqdm

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from dataset.data_modules.data_module import DataModule
from training.losses.diffusion.multisigma_edm_loss import MultisigmaEDMLoss
from training.trainers.diffusion_model_trainer import DiffusionModelTrainer
from utils.distributed.distributed_statistics_accumulator import DistributedStatisticsAccumulator


class SigmaLossPlotter:
    """
    Class for visualizing the loss as a function of the noise level
    """

    def __init__(self):
        pass

    def plot_sigma_loss(self, trainer: DiffusionModelTrainer, config: Dict, dataloader, iterations: int, num_sigmas: int, data_type: str, values_output_file: str, plot_output_file: str, sigma_min: float=0.002, sigma_max: float=160.0):
        """
        :param trainer: The trainer that was used for training
        :param config: The configuration
        :param dataloader: The dataloader to use
        :param iterations: The number of batches to use for plotting
        :param data_type: The type of data to use for plotting. Either "videos" or "images"
        :param values_output_file: The file to which to save the values
        :param plot_output_file: The file to which to save the plot
        """

        model = trainer.ema_model
        device = trainer.device

        if data_type == "videos":
            if not trainer.use_videos:
                raise ValueError("Cannot use videos for plotting if the trainer does not use videos")
            
            batch_adapter = trainer.video_adapter
            masking_strategy = trainer.video_masking_strategy
            
            if "loss" in config["training"]:
                loss_config = config["training"]["loss"]
            else:
                loss_config = config["training"]["video_loss"]

        elif data_type == "images":
            if not trainer.use_images:
                raise ValueError("Cannot use images for plotting if the trainer does not use images")
        
            batch_adapter = trainer.image_adapter
            masking_strategy = trainer.image_masking_strategy

            if "loss" in config["training"]:
                loss_config = config["training"]["loss"]
            else:
                loss_config = config["training"]["image_loss"]

        else:
            raise ValueError("Invalid data type '{}'".format(data_type))

        multisigma_loss_config = {
            "sigma_min": sigma_min,
            "sigma_max": sigma_max,
            "num_sigmas": num_sigmas,

            "edm_loss_config": loss_config
        }

        multisigma_loss = MultisigmaEDMLoss(multisigma_loss_config)

        accumulated_statistics = {}
        with tqdm.tqdm(total=iterations, disable=(dist.get_rank() != 0)) as pbar:
            for batch_idx, current_batch in enumerate(dataloader):
                current_batch.to(device)

                data_entries = batch_adapter.batch_to_data_entries(current_batch)
                masked_data_entries = masking_strategy.apply(data_entries)
                _, current_loss_info = multisigma_loss(model, masked_data_entries)

                # Filters the loss values
                filtered_loss_values = {}
                for current_key in current_loss_info.keys():
                    if "loss_average_sigma_" not in current_key:
                        continue
                    filtered_loss_values[current_key] = current_loss_info[current_key]
                current_loss_info = filtered_loss_values

                for current_key in current_loss_info.keys():
                    if current_key not in accumulated_statistics:
                        accumulated_statistics[current_key] = DistributedStatisticsAccumulator()
                    current_item = current_loss_info[current_key]
                    # (1, 1) Needs to have a batch dimension for the accumulator
                    current_tensor = torch.as_tensor([[current_item]], device=device)
                    accumulated_statistics[current_key].accumulate(current_tensor)
                
                # Synchronizes the processes to avoid they diverge too much during the computation
                torch.distributed.barrier()

                pbar.update(1)
                # Done accumulating statistics
                if batch_idx >= iterations:
                    break
        
        statistics = {}
        for current_key, current_accumulator in accumulated_statistics.items():
            sigma_value = float(current_key.split("_")[-1])
            accumulated_value = current_accumulator.compute_statistics().item()
            statistics[sigma_value] = accumulated_value

        # Saves the values
        with open(values_output_file, "w") as f:
            json.dump(statistics, f, indent=4)

        # Plots the values
        sorted_sigmas = list(sorted(statistics.keys()))
        sorted_values = [statistics[current_sigma] for current_sigma in sorted_sigmas]
        sorted_sigmas = np.asarray(sorted_sigmas)
        sorted_values = np.asarray(sorted_values)

        sns.lineplot(x=sorted_sigmas, y=sorted_values, marker="o", markersize=4, linewidth=1)

        # Checks log scale
        plt.xscale("log")
        # Adds x and y axis labels
        plt.xlabel("Sigma")
        plt.ylabel("Loss")
        plt.ylim(bottom=0.0, top=1.3)
        plt.xlim(sigma_min, sigma_max)
        # Adds the grid
        plt.grid()
        # Saves the plot
        plt.savefig(plot_output_file, bbox_inches='tight')
        plt.close()
import os
from typing import Dict

import wandb
import torch
from lightning.pytorch import Callback
from utils.data_representation.data_entries import DataEntries

from utils.logging.logger import Logger
from utils.visualization.visualization_utils import log_image_data_entries, log_video_data_entries


class GroundTruthVisualizationCallback(Callback):
    def __init__(self, callback_config: Dict, logger: Logger):
        super().__init__()

        self.callback_config = callback_config

        self.logger = logger
        self.logger_prefix = callback_config["logger_prefix"]

        self.every_n_steps = callback_config["log_interval_steps"]

        self.log_videos = callback_config.get("log_videos", True)
        self.log_images = callback_config.get("log_images", True)

        # The adapters to use to extract data from the batch
        if self.log_images:
            image_adapter_config = callback_config['image_batch_adapter']
            image_adapter_target = image_adapter_config['target']
            self.image_adapter = image_adapter_target(image_adapter_config)
        if self.log_videos:
            video_adapter_config = callback_config['video_batch_adapter']
            video_adapter_target = video_adapter_config['target']
            self.video_adapter = video_adapter_target(video_adapter_config)

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        with torch.no_grad():
            if (trainer.global_step % self.every_n_steps == 0) and module.global_rank == 0:
                if self.log_videos:
                    self.visualize_ground_truth_videos(batch)
                if self.log_images:
                    self.visualize_ground_truth_images(batch)

    def visualize_ground_truth_videos(self, batch):
        """
        Logs a batch of ground truth videos to the logger
        :param batch: The batch to log
        """
        # Converts the batch to the network input format
        video_data_entries = self.video_adapter.batch_to_data_entries(batch)
        # Performs logging
        log_video_data_entries(video_data_entries, self.logger, os.path.join(self.logger_prefix, "gt_videos"))

    def visualize_ground_truth_images(self, batch):
        """
        Logs a batch of ground truth images to the logger
        :param batch: The batch to log
        """
        # Converts the batch to the network input format
        image_data_entries = self.image_adapter.batch_to_data_entries(batch)

        log_image_data_entries(image_data_entries, self.logger, os.path.join(self.logger_prefix, "gt_images"))

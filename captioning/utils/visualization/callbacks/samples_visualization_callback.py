import os
from typing import Dict

import wandb
import torch
from lightning.pytorch import Callback
from utils.data_representation.data_entries import DataEntries

from utils.logging.logger import Logger
from utils.visualization.visualization_utils import log_image_data_entries, log_video_data_entries


class SamplesVisualizationCallback(Callback):
    def __init__(self, callback_config: Dict, logger: Logger):
        super().__init__()

        self.callback_config = callback_config

        self.logger = logger
        self.logger_prefix = callback_config["logger_prefix"]

        self.every_n_steps = callback_config["log_interval_steps"]

        self.log_videos = callback_config.get("log_videos", True)
        self.log_images = callback_config.get("log_images", True)
        self.use_ema_if_avaliable = callback_config.get("use_ema_if_avaliable", True)

        if self.log_videos:
            # The strategies to use for masking during sampling
            video_masking_strategy_config = self.callback_config['video_masking_strategy']
            video_masking_strategy_target = video_masking_strategy_config['target']
            self.video_masking_strategy = video_masking_strategy_target(video_masking_strategy_config)
            # The adapters to use to extract data from the batch
            video_adapter_config = callback_config['video_batch_adapter']
            video_adapter_target = video_adapter_config['target']
            self.video_adapter = video_adapter_target(video_adapter_config)

        if self.log_images:
            # Does the same for images
            image_masking_strategy_config = self.callback_config['image_masking_strategy']
            image_masking_strategy_target = image_masking_strategy_config['target']
            self.image_masking_strategy = image_masking_strategy_target(image_masking_strategy_config)
            image_adapter_config = callback_config['image_batch_adapter']
            image_adapter_target = image_adapter_config['target']
            self.image_adapter = image_adapter_target(image_adapter_config)

        # Instantiates the sampler
        sampler_config = self.callback_config["sampler"]
        sampler_target = sampler_config["target"]
        self.sampler = sampler_target(sampler_config)

        self.forced_class_label = callback_config.get("forced_class_label", None)

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        with torch.no_grad():
            if (trainer.global_step % self.every_n_steps == 0):

                if self.log_videos:
                    self.visualize_generated_videos(module, batch)
                if self.log_images:
                    self.visualize_generated_images(module, batch)

    def visualize_generated_videos(self, module, batch):
        """
        Logs a visualization of sampled generated video entries
        :param module: The module to use for sampling
        :param batch: The batch on which to build the samples. Defines conditioning information
        """
        model = module.model
        if self.use_ema_if_avaliable and hasattr(module, "ema_model"):
            model = module.ema_model

        # Converts the batch to the network input format
        video_data_entries = self.video_adapter.batch_to_data_entries(batch)
        # Masks the video
        masked_video_data_entries = self.video_masking_strategy.apply(video_data_entries)
        if self.forced_class_label is not None:
            masked_video_data_entries["class_labels"].data = (masked_video_data_entries["class_labels"].data * 0 + self.forced_class_label).long()
        # Performs sampling
        sampled_video_data_entries = self.sampler(model, masked_video_data_entries)
        # Logs only on the first process.
        # NOTE: Generation needs to happen on all processes for FSDP to work
        if module.global_rank == 0:
            log_video_data_entries(sampled_video_data_entries, self.logger, os.path.join(self.logger_prefix, "sampled_videos"))

    def visualize_generated_images(self, module, batch):
        """
        Logs a visualization of sampled generated image entries
        :param module: The module to use for sampling
        :param batch: The batch on which to build the samples. Defines conditioning information
        """
        model = module.model
        if self.use_ema_if_avaliable and hasattr(module, "ema_model"):
            model = module.ema_model

        # Converts the batch to the network input format
        image_data_entries = self.image_adapter.batch_to_data_entries(batch)
        # Masks the video
        masked_image_data_entries = self.image_masking_strategy.apply(image_data_entries)
        if self.forced_class_label is not None:
            masked_image_data_entries["class_labels"].data = (masked_image_data_entries["class_labels"].data * 0 + self.forced_class_label).long()
        # Performs sampling
        sampled_image_data_entries = self.sampler(model, masked_image_data_entries)

        # Logs only on the first process.
        # NOTE: Generation needs to happen on all processes for FSDP to work
        if module.global_rank == 0:
            log_image_data_entries(sampled_image_data_entries, self.logger, os.path.join(self.logger_prefix, "sampled_images"))


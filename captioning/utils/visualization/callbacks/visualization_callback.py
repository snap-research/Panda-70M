import os
from typing import Dict

import wandb
import torch
from lightning.pytorch import Callback


class VisualizationCallback(Callback):
    def __init__(self, callback_config: Dict):
        super().__init__()

        self.callback_config = callback_config
        self.every_n_steps = callback_config["every_n_steps"]

        # TODO get the masking strategy
        # TODO get the sampler
        # TODO create a prefix for the logger since we may have multiple ones

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        # TODO need to redo
        with torch.no_grad():
            if (trainer.global_step % self.every_n_steps == 0) and module.global_rank == 0:
                logger = module.logger.experiment
                self.visualize_ground_truth_videos(logger, batch)
                self.visualize_ground_truth_images(logger, batch)
                self.visualize_generated_videos(module, logger, batch)
                self.visualize_generated_images(module, logger, batch)

    def visualize_ground_truth_videos(self, logger, batch):
        # TODO need to redo
        columns = ["Video", "Text", "Class Label"]
        table_data = []
        video = batch['video_data'].permute(0, 2, 1, 3, 4).cpu().numpy()
        video = (video * 127.5 + 127.5).astype('uint8')
        for i in range(video.shape[0]):
            table_data.append([wandb.Video(video[i], fps=4, format="gif"),
                               batch['video_text'][i],
                               str(batch['video_cls'][i].cpu().numpy())])
        table = wandb.Table(data=table_data, columns=columns)
        logger.log({"gt_video": table})

    def visualize_ground_truth_images(self, logger, batch):
        # TODO need to redo
        table_data = []
        columns = ["Image", "Text", "Class Label"]

        x = batch['image_data']
        x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])
        labels = batch['image_cls']
        labels = labels.view(-1)
        text = sum(batch['image_text'], [])
        image = (x.permute(0, 2, 3, 1).cpu().numpy() * 127.5 + 127.5).astype('uint8')

        for i in range(x.shape[0]):
            table_data.append([wandb.Image(image[i]),
                               text[i],
                               str(labels[i].cpu().numpy())])
        table = wandb.Table(data=table_data, columns=columns)
        logger.log({"gt_image": table})

    def visualize_generated_videos(self, module, logger, batch):
        # TODO need to redo
        random_video_batch = module.sample_videos(class_labels=batch['video_cls'])
        random_video_batch = random_video_batch.data.permute(0, 2, 1, 3, 4).cpu().numpy()

        table_data = []
        columns = ["Video", "Text", "Class Label"]
        random_video_batch = (random_video_batch * 127.5 + 127.5).astype('uint8')

        for i in range(random_video_batch.shape[0]):
            table_data.append([wandb.Video(random_video_batch[i], fps=4, format="gif"),
                               batch['video_text'][i],
                               str(batch['video_cls'][i].cpu().numpy())])
        table = wandb.Table(data=table_data, columns=columns)
        logger.log({"random_video": table})

    def visualize_generated_images(self, module, logger, batch):
        # TODO need to redo
        x = batch['image_data']
        x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])
        labels = batch['image_cls']
        labels = labels.view(-1)
        text = sum(batch['image_text'], [])

        random_image_batch = module.sample_images(class_labels=labels)
        random_image_batch = random_image_batch.data.permute(0, 2, 3, 1).cpu().numpy()
        random_image_batch = (random_image_batch * 127.5 + 127.5).astype('uint8')

        table_data = []
        columns = ["Image", "Text", "Class Label"]
        for i in range(random_image_batch.shape[0]):
            table_data.append([wandb.Image(random_image_batch[i]),
                               text[i],
                               str(labels[i].cpu().numpy())])
        table = wandb.Table(data=table_data, columns=columns)
        logger.log({"random_image": table})



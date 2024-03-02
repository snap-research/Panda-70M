import os
from pathlib import Path
from typing import Dict

from PIL import Image
import einops
import wandb
import torch
import numpy as np
import json
from lightning.pytorch import Callback
from utils.data_representation.data_entries import DataEntries

from utils.logging.logger import Logger
from utils.video.video_encoder import VideoEncoder

def log_video_data_entries(data_entries: DataEntries, logger: Logger, log_name: str, video_key: str="input", lowres_key: str="input_lowres",
                          class_labels_key: str="class_labels", summary_text_key: str="summary_text"):
    """
    Logs the videos in the given data entries
    :param data_entries: The data entries to log
    :param logger: The logger to use
    :param log_name: The name to assign to the logged data
    :param video_key: The key of the video in the data entries
    :param class_labels_key: The key of the class labels in the data entries
    :param summary_text_key: The key of the summary text in the data entries
    """
    # (batch_size, frames_count, channels, height, width)
    video = data_entries[video_key].data
    video_numpy = video.cpu().numpy()

    if lowres_key in data_entries.keys():
        # (batch_size, frames_count, channels, height, width)
        video_lowres = data_entries[lowres_key].data
        video_lowres_numpy = video_lowres.cpu().numpy()
        video_lowres_numpy = np.clip((video_lowres_numpy * 127.5 + 128), 0, 255).astype('uint8')
    else:
        video_lowres_numpy = None

    batch_size = video_numpy.shape[0]
    # (batch_size)
    class_labels_numpy = np.zeros(batch_size, dtype=np.int32)
    if class_labels_key in data_entries.keys():
        class_labels_numpy = data_entries[class_labels_key].data.cpu().numpy()
    # (batch_size)
    summary_text = [""] * batch_size
    if summary_text_key in data_entries.keys():
        summary_text = data_entries[summary_text_key].data

    columns = (["Video Low"] if video_lowres_numpy is not None else []) + ["Video", "Summary Text", "Class Label"]

    table_data = []
    video_numpy = np.clip((video_numpy * 127.5 + 128), 0, 255).astype('uint8')
    for i in range(batch_size):
        if video_lowres_numpy is not None:
            wandb_video_lowres = [wandb.Video(video_lowres_numpy[i], fps=4, format="mp4")]
        else:
            wandb_video_lowres = []
        table_data.append(wandb_video_lowres + [wandb.Video(video_numpy[i], fps=4, format="mp4"), summary_text[i], str(class_labels_numpy[i])])
    table = wandb.Table(data=table_data, columns=columns)
    logger.get_logger().experiment.log({f"{log_name}": table})

def log_image_data_entries(data_entries: DataEntries, logger: Logger, log_name: str, image_key: str="input", lowres_key: str="input_lowres",
                           class_labels_key: str="class_labels", summary_text_key: str="summary_text"):
    """
    Logs the images in the given data entries
    :param data_entries: The data entries to log
    :param logger: The logger to use
    :param log_name: The name to assign to the logged data
    :param image_key: The key of the video in the data entries
    :param class_labels_key: The key of the class labels in the data entries
    :param summary_text_key: The key of the summary text in the data entries
    """
    # (batch_size, channels, height, width)
    images = data_entries[image_key].data
    # Some models may represent images as a video with a sequence of frames (batch_size, frames_count, channels, height, width). We extract the first frame
    if len(images.shape) == 5:
        images = images[:, 0]

    images_numpy = images.cpu().numpy()
    batch_size = images_numpy.shape[0]

    if lowres_key in data_entries.keys():
        # (batch_size, frames_count, channels, height, width)
        image_lowres = data_entries[lowres_key].data
        image_lowres_numpy = image_lowres.cpu().numpy()
        if len(image_lowres_numpy.shape) == 5:
            image_lowres_numpy = image_lowres_numpy[:, 0]
        image_lowres_numpy = np.clip((image_lowres_numpy * 127.5 + 128), 0, 255).astype('uint8')
        image_lowres_numpy = np.moveaxis(image_lowres_numpy, 1, -1)
    else:
        image_lowres_numpy = None

    # (batch_size)
    class_labels_numpy = np.zeros(batch_size, dtype=np.int32)
    if class_labels_key in data_entries.keys():
        class_labels_numpy = data_entries[class_labels_key].data.cpu().numpy()

    # (batch_size)
    summary_text = [""] * batch_size
    if summary_text_key in data_entries.keys():
        summary_text = data_entries[summary_text_key].data

    table_data = []
    columns =  (["Image Low"] if image_lowres_numpy is not None else []) + ["Image", "Summary Text", "Class Label"]
    images_numpy = np.clip((images_numpy * 127.5 + 128), 0, 255).astype('uint8')
    # (batch_size, height, width, channels)
    images_numpy = np.moveaxis(images_numpy, 1, -1)
    for i in range(batch_size):
        if image_lowres_numpy is not None:
            wandb_image_lowres = [wandb.Image(image_lowres_numpy[i])]
        else:
            wandb_image_lowres = []
        table_data.append(wandb_image_lowres + [wandb.Image(images_numpy[i]), summary_text[i], str(class_labels_numpy[i])])
    table = wandb.Table(data=table_data, columns=columns)
    logger.get_logger().experiment.log({f"{log_name}": table})

def log_image_data_entries_to_disk(data_entries: DataEntries, directory: str, output_prefix: str, image_key: str="input", class_labels_key: str="class_labels", summary_text_key: str="summary_text", gt_key: str="input_gt", limit: int=None):
    """
    Saves the images in the given data entries to disk
    :param data_entries: The data entries to log
    :param directory: The directory where to save the images
    :param output_prefix: The prefix to use for the output files
    :param image_key: The key of the video in the data entries
    :param class_labels_key: The key of the class labels in the data entries
    :param summary_text_key: The key of the summary text in the data entries
    :param limit: The maximum number of images to save
    """
    Path(directory).mkdir(parents=True, exist_ok=True)

    # (batch_size, channels, height, width)
    images = data_entries[image_key].data
    # Some models may represent images as a video with a sequence of frames (batch_size, frames_count, channels, height, width). We extract the first frame
    if len(images.shape) == 5:
        images = images[:, 0]

    images_numpy = images.cpu().numpy()
    batch_size = images_numpy.shape[0]
    images_numpy = np.clip((images_numpy * 127.5 + 128), 0, 255).astype('uint8')
    # (batch_size, height, width, channels)
    images_numpy = np.moveaxis(images_numpy, 1, -1)

    if gt_key in data_entries.keys():
        # (batch_size, frames_count, channels, height, width)
        image_gt = data_entries[gt_key].data
        image_gt_numpy = image_gt.cpu().numpy()
        if len(image_gt_numpy.shape) == 5:
            image_gt_numpy = image_gt_numpy[:, 0]
        image_gt_numpy = np.clip((image_gt_numpy * 127.5 + 128), 0, 255).astype('uint8')
        image_gt_numpy = np.moveaxis(image_gt_numpy, 1, -1)
        images_numpy = np.concatenate((images_numpy, image_gt_numpy), axis=1)
    else:
        image_gt_numpy = None

    # (batch_size)
    class_labels_numpy = np.zeros(batch_size, dtype=np.int32) - 1
    if class_labels_key in data_entries.keys():
        class_labels_numpy = data_entries[class_labels_key].data.cpu().numpy()

    # (batch_size)
    summary_text = [""] * batch_size
    has_summary_text = summary_text_key in data_entries.keys()
    if has_summary_text:
        summary_text = data_entries[summary_text_key].data

    table_data = []

    for i in range(batch_size):
        if limit is not None and i >= limit:
            break

        current_image = Image.fromarray(images_numpy[i])
        current_image_name = f"{output_prefix}_{i:05d}"
        current_class_label = int(class_labels_numpy[i])
        if current_class_label >= 0:
            current_image_name += f"_class_{current_class_label:04d}"
        current_image_name += ".png"
        current_image.save(os.path.join(directory, current_image_name))

        current_summary_text = summary_text[i]
        current_metadata_filename = os.path.splitext(current_image_name)[0] + ".metadata.json"
        metadata = {}
        # Saves the summary text only if present in the original data entries
        if has_summary_text:
            metadata["summary_text"] = current_summary_text
        # Saves the medatata only if not empty to avoid creation of excessive number of files
        if len(metadata.keys()) > 0:
            with open(os.path.join(directory, current_metadata_filename), "w") as metadata_file:
                json.dump(metadata, metadata_file)

        # Checks if per-sample-step results are present in the data entries and saves them to disk if so
        if "sampler_sigmas" in data_entries.keys():
            # (batch_size, sample_steps) list with the sigmas employed for producing each intermediate sample
            current_sampler_sigmas = data_entries["sampler_sigmas"].data[i]
            all_sampler_step_images = []
            for current_sigma_index, current_sigma in enumerate(current_sampler_sigmas):
                current_sampler_step_image_key = f"{image_key}_denoised_sigma_{current_sigma:.4f}"
                if current_sampler_step_image_key in data_entries.keys():
                    current_sampler_step_image = data_entries[current_sampler_step_image_key].data
                    current_sampler_step_image = current_sampler_step_image[i]
                    if len(current_sampler_step_image.shape) == 4: # If the current batch element is a video (frames_count, channels, height, width)
                        # Stacks the frames on the height dimension
                        current_sampler_step_image = einops.rearrange(current_sampler_step_image, "f c h w -> c (f h) w")
                    all_sampler_step_images.append(current_sampler_step_image)

            all_sampler_step_images = torch.cat(all_sampler_step_images, dim=-1) # Stacks along the width dimension
            all_sampler_step_images_numpy = all_sampler_step_images.cpu().numpy()
            all_sampler_step_images_numpy = np.clip((all_sampler_step_images_numpy * 127.5 + 128), 0, 255).astype('uint8')
            # (height, width * sampler_steps, channels)
            all_sampler_step_images_numpy = np.moveaxis(all_sampler_step_images_numpy, 0, -1)
            all_sampler_step_images_pil = Image.fromarray(all_sampler_step_images_numpy)
            sampler_step_image_name = f"{output_prefix}_{i:05d}"
            if current_class_label >= 0:
                sampler_step_image_name += f"_class_{current_class_label:04d}"
            sampler_step_image_name += "_sampler_steps.png"
            sampler_step_image_path = os.path.join(directory, sampler_step_image_name)

            # Saves the images
            all_sampler_step_images_pil.save(sampler_step_image_path)

def log_video_data_entries_to_disk(data_entries: DataEntries, directory: str, output_prefix: str, framerate: float, video_key: str="input", class_labels_key: str="class_labels", summary_text_key: str="summary_text", gt_key: str="input_gt", limit: int=None):
    """
    Saves the videos in the given data entries to disk
    :param data_entries: The data entries to log
    :param directory: The directory where to save the videos
    :param output_prefix: The prefix to use for the output files
    :param video_key: The key of the video in the data entries
    :param class_labels_key: The key of the class labels in the data entries
    :param summary_text_key: The key of the summary text in the data entries
    :param limit: The maximum number of videos to save
    """
    Path(directory).mkdir(parents=True, exist_ok=True)

    # (batch_size, channels, frames, height, width)
    videos = data_entries[video_key].data
    videos_numpy = videos.cpu().numpy()
    batch_size = videos_numpy.shape[0]
    # (batch_size)
    class_labels_numpy = np.zeros(batch_size, dtype=np.int32) - 1
    if class_labels_key in data_entries.keys():
        class_labels_numpy = data_entries[class_labels_key].data.cpu().numpy()
    
    if gt_key in data_entries.keys():
        # (batch_size, frames_count, channels, height, width)
        video_gt = data_entries[gt_key].data
        video_gt_numpy = video_gt.cpu().numpy()
        video_gt_numpy = np.clip((video_gt_numpy * 127.5 + 128), 0, 255).astype('uint8')
        video_gt_numpy = np.moveaxis(video_gt_numpy, 2, -1)
    else:
        video_gt_numpy = None

    # (batch_size)
    summary_text = [""] * batch_size
    has_summary_text = summary_text_key in data_entries.keys()
    if has_summary_text:
        summary_text = data_entries[summary_text_key].data

    for i in range(batch_size):
        if limit is not None and i >= limit:
            break

        current_video = videos_numpy[i]
        current_video = np.clip((current_video * 127.5 + 128), 0, 255).astype('uint8')
        # (frames, height, width, channels)
        current_video = np.moveaxis(current_video, 1, -1)

        if video_gt_numpy is not None:
            current_video = np.concatenate((current_video, video_gt_numpy[i]), axis=2)

        current_video_name = f"{output_prefix}_{i:05d}"
        current_class_label = int(class_labels_numpy[i])
        if current_class_label >= 0:
            current_video_name += f"_class_{current_class_label:04d}"
        current_video_name += ".mp4"
        current_video_path = os.path.join(directory, current_video_name)

        encoding_parameters = {
            "framerate": framerate
        }
        video_encoder = VideoEncoder()
        video_encoder.encode_numpy_video(current_video, current_video_path, encoding_parameters)
            
        current_summary_text = summary_text[i]
        current_metadata_filename = os.path.splitext(current_video_name)[0] + ".metadata.json"
        metadata = {}
        # Saves the summary text only if present in the original data entries
        if has_summary_text:
            metadata["summary_text"] = current_summary_text
        # Saves the medatata only if not empty to avoid creation of excessive number of files
        if len(metadata.keys()) > 0:
            with open(os.path.join(directory, current_metadata_filename), "w") as metadata_file:
                json.dump(metadata, metadata_file)

        # Checks if per-sample-step results are present in the data entries and saves them to disk if so
        if "sampler_sigmas" in data_entries.keys():
            # (batch_size, sample_steps) list with the sigmas employed for producing each intermediate sample
            current_sampler_sigmas = data_entries["sampler_sigmas"].data[i]
            all_sampler_step_videos = []
            for current_sigma_index, current_sigma in enumerate(current_sampler_sigmas):
                current_sampler_step_video_key = f"{video_key}_denoised_sigma_{current_sigma:.4f}"
                if current_sampler_step_video_key in data_entries.keys():
                    current_sampler_step_video = data_entries[current_sampler_step_video_key].data[i]
                    all_sampler_step_videos.append(current_sampler_step_video)
            all_sampler_step_videos = torch.cat(all_sampler_step_videos, dim=-1) # Stacks along the width dimension
            all_sampler_step_videos_numpy = all_sampler_step_videos.cpu().numpy()
            all_sampler_step_videos_numpy = np.clip((all_sampler_step_videos_numpy * 127.5 + 128), 0, 255).astype('uint8')
            # (frames, height, width * sampler_steps, channels)
            all_sampler_step_videos_numpy = np.moveaxis(all_sampler_step_videos_numpy, 1, -1)
            sampler_step_video_name = f"{output_prefix}_{i:05d}"
            if current_class_label >= 0:
                sampler_step_video_name += f"_class_{current_class_label:04d}"
            sampler_step_video_name += "_sampler_steps.mp4"
            sampler_step_video_path = os.path.join(directory, sampler_step_video_name)

            # Saves the video
            video_encoder.encode_numpy_video(all_sampler_step_videos_numpy, sampler_step_video_path, encoding_parameters)

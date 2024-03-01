from typing import Dict

from utils.data_representation.batch_adapters.batch_adapter import BatchAdapter
from utils.data_representation.data_entries import DataEntries
from utils.data_representation.entries.data_entry import DataEntry

from torch.nn import functional as F
import torch
from torch import meshgrid

def random_crop_batch(images, crop_size):
    """
    Randomly crop a batch of images.

    Args:
        images (torch.Tensor): Batch of input images with shape (batch_size, channels, height, width).
        crop_size (tuple): Tuple containing the desired crop size (crop_height, crop_width).

    Returns:
        torch.Tensor: Batch of randomly cropped images with the same shape as input images.
    """

    is_video = (len(images.shape) == 5)
    if is_video:
        batch_size, channels, time, height, width = images.size()
        crop_time, crop_height, crop_width = crop_size
    else:
        batch_size, channels, height, width = images.size()
        crop_height, crop_width = crop_size

    if crop_height > height or crop_width > width or (is_video and crop_time > time):
        raise ValueError("Crop size must be smaller than the input image size")

    # Generate random crop positions for each image in the batch
    top = torch.randint(0, height - crop_height + 1, (batch_size, 1), device=images.device)
    left = torch.randint(0, width - crop_width + 1, (batch_size, 1), device=images.device)
    if is_video:
        start = torch.randint(0, time - crop_time + 1, (batch_size, 1), device=images.device)

    if is_video:
        cropped_images = torch.empty((batch_size, channels, crop_time, crop_height, crop_width), dtype=images.dtype, device=images.device)
    else:
        cropped_images = torch.empty((batch_size, channels, crop_height, crop_width), dtype=images.dtype, device=images.device)

    # Crop each image in the batch
    for i in range(batch_size):
        if is_video:
            cropped_images[i] = images[i, :, start[i]:start[i] + crop_time, top[i]:top[i] + crop_height, left[i]:left[i] + crop_width]
        else:
            cropped_images[i] = images[i, :, top[i]:top[i] + crop_height, left[i]:left[i] + crop_width]

    
    return cropped_images


class RandomCrop:
    """
    Class that takes original data entries and append additional lowres field into it.
    """
    def __init__(self, config: Dict):
        crop_size = config.get("crop_size", (72, 128))
        self.crop_size = crop_size
        
    def __call__(self, data_entries: DataEntries) -> DataEntries:
        data_entries = data_entries.shallow_copy()
        
        for current_input_key in list(data_entries.keys(types_filter="input")):
            current_data_entry = data_entries[current_input_key]
            current_input = current_data_entry.data 
            new_input = random_crop_batch(current_input, self.crop_size)
            data_entries.add(DataEntry(current_input_key, new_input, type="input"))

        return data_entries


if __name__ == "__main__":
    import numpy as np
    images = torch.Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    print (images)
    images = images.unsqueeze(0).unsqueeze(0)

    print (random_crop_batch(images, (1, 2)))


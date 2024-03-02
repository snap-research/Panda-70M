from typing import Dict, Any

import torch

from utils.data_representation.batch_adapters.batch_adapter import BatchAdapter
from utils.data_representation.data_entries import DataEntries
from utils.data_representation.entries.data_entry import DataEntry

class ParallelIterationImageBatchAdapter(BatchAdapter):
    """
    Class representing the batch adapter between parallel image/video iteration batch format and the base EDM model
    """
    def __init__(self, adapter_config: Dict):
        super().__init__(adapter_config)

    def batch_to_data_entries(self, batch: Any) -> DataEntries:
        """
        Converts a batch to the corresponding data entries
        :param batch: The batch to convert
        :return: The batch converted to data entries
        """
        image_data = batch.data['image']

        # (batch_size, features, height, width) or (batch_size, frames_count, features, height, width) depending on the structure of the image model
        images = image_data["video"]

        data_entries = DataEntries()
        data_entries.add(DataEntry("input", images, type="input"))
        if "class" in image_data:
            class_labels = image_data["class"]
            data_entries.add(DataEntry("class_labels", class_labels))

        if "summary_text" in image_data:
            summary_text = image_data["summary_text"]
            data_entries.add(DataEntry("summary_text", summary_text))

        # Adds text embeddings if present
        if "summary_text_embeddings" in image_data:
            summary_text_embeddings = image_data["summary_text_embeddings"]
            data_entries.add(DataEntry("summary_text_embeddings", summary_text_embeddings))

        # Builds the labels indicating whether the media is an image or a video
        # Image = 0, Video = 1
        video_image_labels = torch.zeros(images.shape[0], dtype=torch.long, device=images.device)
        data_entries.add(DataEntry("video_image_labels", video_image_labels))

        # Gets information on the resolution if available
        if "resolution" in image_data:
            resolution = image_data["resolution"]
            data_entries.add(DataEntry("resolution", resolution))


        # Gets the information on the dataset id if available
        if "dataset_id" in image_data:
            dataset_id = image_data["dataset_id"]
            data_entries.add(DataEntry("dataset_id", dataset_id))

        # Gets the information on the sampling framerate if available
        if "sampling_framerate" in image_data:
            sampling_framerate = image_data["sampling_framerate"]
            data_entries.add(DataEntry("sampling_framerate", sampling_framerate))

        return self.transform(data_entries)


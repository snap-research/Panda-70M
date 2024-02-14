from typing import Dict, Any

from utils.data_representation.batch_adapters.batch_adapter import BatchAdapter
from utils.data_representation.data_entries import DataEntries
from utils.data_representation.entries.data_entry import DataEntry


class KineticsImageBatchAdapter(BatchAdapter):
    """
    Class representing the batch adapter between Alex's Kinetics dataset format and the base EDM model
    """
    def __init__(self, adapter_config: Dict):
        super().__init__(adapter_config)

    def batch_to_data_entries(self, batch: Any) -> DataEntries:
        """
        Converts a batch to the corresponding data entries
        :param batch: The batch to convert
        :return: The batch converted to data entries
        """
        images = batch['image_data']
        # (batch_size, features, height, width)
        images = images.view(-1, images.shape[2], images.shape[3], images.shape[4])
        image_labels = batch['image_cls']
        image_labels = image_labels.view(-1)

        data_entries = DataEntries()
        data_entries.add(DataEntry("input", images, type="input"))
        data_entries.add(DataEntry("class_labels", image_labels))

        return data_entries


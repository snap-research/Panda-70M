from typing import Dict, Any

from utils.data_representation.batch_adapters.batch_adapter import BatchAdapter
from utils.data_representation.data_entries import DataEntries
from utils.data_representation.entries.data_entry import DataEntry


class KineticsVideoBatchAdapter(BatchAdapter):
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
        videos = batch['video_data']
        video_labels = batch['video_cls']

        data_entries = DataEntries()
        data_entries.add(DataEntry("input", videos, type="input"))
        data_entries.add(DataEntry("class_labels", video_labels))

        return data_entries


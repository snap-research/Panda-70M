from typing import Dict, List
import torch

from utils.data_representation.data_entries import DataEntries
from utils.data_representation.entries.data_entry import DataEntry
from utils.data_representation.masking.masking_strategy import MaskingStrategy


class IdentityMaskingStrategy(MaskingStrategy):
    """
    A masking strategy that does nothing
    """
    def __init__(self, masking_config: Dict):
        super().__init__(masking_config)

    def apply(self, data_entries: DataEntries, batch_ids: List[int]=None) -> DataEntries:
        """
        Applies the masking strategy to the data
        :param data_entries: the data to which to apply masking
        :param batch_ids: the batch element ids to which to apply the masking. If None applies to all batch elements
        :return: the data entries with masking applied
        """
        inferred_batch_size = data_entries.infer_batch_size()
        inferred_device = data_entries.infer_device()

        # Creates new entries by assigning them the default mask
        new_data_entries = DataEntries()
        for data_entry in data_entries.values():
            new_data_entry = self.create_default_mask(data_entry, inferred_batch_size, inferred_device)
            new_data_entries.add(new_data_entry)

        return new_data_entries

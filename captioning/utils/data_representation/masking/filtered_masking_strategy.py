from typing import Dict, List

from utils.data_representation.data_entries import DataEntries
from utils.data_representation.entries.data_entry import DataEntry
from utils.data_representation.masking.masking_strategy import MaskingStrategy


class FilteredMaskingStrategy(MaskingStrategy):
    """
    Base strategy to apply masking that applies it only to a selection of the data entries
    """
    def __init__(self, masking_config: Dict):
        super().__init__(masking_config)

        self.keys_filter = masking_config.get("keys_filter", None)
        self.types_filter = masking_config.get("types_filter", None)

    def apply(self, data_entries: DataEntries, batch_ids: List[int]=None) -> DataEntries:
        """
        Applies the masking strategy to the data
        :param data_entries: the data to which to apply masking
        :param batch_ids: the batch element ids to which to apply the masking. If None applies to all batch elements
        :return: The data entries with the masking applied
        """
        def apply_masking(data_entry: DataEntry, data_entries: DataEntries):
            return self.apply_to_entry(data_entry, batch_ids, data_entries)
        # Applies masking only to the entries matching the selection criteria
        masked_data_entries = data_entries.apply(apply_masking, keys_filter=self.keys_filter, types_filter=self.types_filter)
        return masked_data_entries

    def apply_to_entry(self, data_entry: DataEntry, batch_ids: List[int], data_entries: DataEntries) -> DataEntry:
        """
        Abstract method that applies the masking strategy to the data entry.
        The received data entry is a shallow copy of the original data entry, so it is safe to assign different values to its attributes.
        :param data_entry: the data entry to which to apply masking
        :param batch_ids: the batch element ids to which to apply the masking. If None applies to all batch elements
        :param data_entries: the data entries to which the data entry belongs
        :return: The data entry with the masking applied
        """
        raise NotImplementedError

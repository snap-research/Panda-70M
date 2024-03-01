from typing import Dict, List

import torch
import torch.nn as nn

from utils.data_representation.data_entries import DataEntries
from utils.data_representation.entries.data_entry import DataEntry
from utils.data_representation.masking.masking_strategy import MaskingStrategy


class SynchronizedBernoulliBatchMaskingStrategy(MaskingStrategy):
    """
    Mask strategy to apply masking that applies masking in the batch dimension independently to each batch element with a certain probability.
    If multiples fields match the filters, applies the same mask to all elements
    """
    def __init__(self, masking_config: Dict):
        super().__init__(masking_config)

        self.keys_filter = masking_config.get("keys_filter", None)
        self.types_filter = masking_config.get("types_filter", None)

        self.masking_probability = masking_config["masking_probability"]

    def apply(self, data_entries: DataEntries, batch_ids: List[int]=None) -> DataEntries:
        """
        Applies the masking strategy to the data
        :param data_entries: the data to which to apply masking
        :param batch_ids: the batch element ids to which to apply the masking. If None applies to all batch elements
        :return: The data entries with the masking applied
        """

        synchronized_mask = None

        masked_data_entries = data_entries.shallow_copy()
        for current_entry_key in list(data_entries.keys(keys_filter=self.keys_filter, types_filter=self.types_filter)):
            
            current_data_entry = masked_data_entries[current_entry_key]
            batch_size = current_data_entry.infer_batch_size()  # The batch size is supposed to be inferrable from a single data entry and may vary in different entries
            device = data_entries.infer_device()  # The device is supposed to be the same for all data entries, so we infer it from there

            # Creates the default mask of size (batch_size)
            masked_data_entry = self.create_default_mask(current_data_entry, batch_size, device)

            # Creates the common mask
            if synchronized_mask is None:
                synchronized_mask = torch.rand((batch_size,), device=device) > self.masking_probability  # Puts to False=masked with the given probability

                # Computes the batch ids where the mask should not be applied
                batch_ids_to_preserve = self.compute_batch_ids_to_preserve(batch_size, batch_ids)
                synchronized_mask[batch_ids_to_preserve] = True  # Preserve the elements that are not to be masked
            
            # Uses to common mask    
            current_mask = synchronized_mask.clone()

            # Makes the current mask of a dimension compatible with the existing mask
            current_mask = current_mask.reshape([current_mask.shape[0]] + [1] * (masked_data_entry.mask.ndim - 1))

            masked_data_entry.mask = torch.logical_and(masked_data_entry.mask, current_mask)  # Applies the mask to the default mask
            masked_data_entries[current_entry_key] = masked_data_entry

        return masked_data_entries

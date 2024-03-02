from typing import Dict, List

import torch

from utils.data_representation.data_entries import DataEntries
from utils.data_representation.entries.data_entry import DataEntry
from utils.data_representation.masking.filtered_masking_strategy import FilteredMaskingStrategy
from utils.data_representation.masking.masking_strategy import MaskingStrategy


class BernoulliBatchMaskingStrategy(FilteredMaskingStrategy):
    """
    Mask strategy to apply masking that applies masking in the batch dimension independently to each batch element with a certain probability
    """
    def __init__(self, masking_config: Dict):
        super().__init__(masking_config)

        self.masking_probability = masking_config["masking_probability"]

    def apply_to_entry(self, data_entry: DataEntry, batch_ids: List[int], data_entries: DataEntries) -> DataEntry:
        """
        Abstract method that applies the masking strategy to the data entry.
        The received data entry is a shallow copy of the original data entry, so it is safe to assign different values to its attributes.
        :param data_entry: the data entry to which to apply masking
        :param batch_ids: the batch element ids to which to apply the masking. If None applies to all batch elements
        :param data_entries: the data entries to which the data entry belongs
        :return: The data entry with the masking applied
        """
        batch_size = data_entry.infer_batch_size()  # The batch size is supposed to be inferrable from a single data entry and may vary in different entries
        device = data_entries.infer_device()  # The device is supposed to be the same for all data entries, so we infer it from there

        # Creates the default mask of size (batch_size)
        masked_data_entry = self.create_default_mask(data_entry, batch_size, device)

        current_mask = torch.rand((batch_size,), device=device) > self.masking_probability  # Puts to False=masked with the given probability

        # Computes the batch ids where the mask should not be applied
        batch_ids_to_preserve = self.compute_batch_ids_to_preserve(batch_size, batch_ids)
        current_mask[batch_ids_to_preserve] = True  # Preserve the elements that are not to be masked
        # Makes the current mask of a dimension compatible with the existing mask
        current_mask = current_mask.reshape([current_mask.shape[0]] + [1] * (masked_data_entry.mask.ndim - 1))

        masked_data_entry.mask = torch.logical_and(masked_data_entry.mask, current_mask)  # Applies the mask to the default mask

        return masked_data_entry


def main():

    import torch
    import numpy as np

    batch_size = 16

    data_entries = DataEntries()
    data_entries.add(DataEntry("key1", ["value1"] * batch_size, type="type1"))
    data_entries.add(DataEntry("key2", torch.zeros((batch_size, 4)), type="type2"))
    data_entries.add(DataEntry("key3", ["value1"] * batch_size, torch.ones((batch_size, 4), dtype=torch.bool), type="type1"))
    data_entries.add(DataEntry("key4", torch.zeros((batch_size, 4)), torch.ones((batch_size), dtype=torch.bool), type="type1"))
    data_entries.add(DataEntry("key5", np.zeros((batch_size, 4)), type="type2"))
    data_entries.add(DataEntry("key6", ["value1"] * batch_size, torch.zeros((batch_size, 4), dtype=torch.bool), type="type1"))
    data_entries.add(DataEntry("key7", np.zeros((batch_size, 4)), torch.zeros((batch_size), dtype=torch.bool), type="type1"))
    data_entries["key8"] = DataEntry("key8", ["value1"] * batch_size)

    print(data_entries)

    masking_strategy_config = {
        #"keys_filter": "key1",
        "masking_probability": 0.3,
    }

    masking_strategy = BernoulliBatchMaskingStrategy(masking_strategy_config)
    masked_data_entries = masking_strategy.apply(data_entries, batch_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8])

    print(masked_data_entries)


if __name__ == "__main__":
    main()

from typing import Dict, List
import torch

from utils.data_representation.data_entries import DataEntries
from utils.data_representation.entries.data_entry import DataEntry


class MaskingStrategy:
    """
    Base class representing a strategy with which to apply masking to the data
    Uses the following convention for mask values:
        - True: the element in the corresponding position is valid (not masked)
        - False: the element in the corresponding position is masked
    """
    def __init__(self, masking_config: Dict):
        self.masking_config = masking_config

    def create_default_mask(self, data_entry: DataEntry, batch_size: int=None, device=None) -> DataEntry:
        """
        Creates a data entry with a default mask applied
        :param data_entry: The entry for which to create the mask
        :param batch_size: hint on the batch size to use
        :param device: hint on the device to use
        :return:
        """
        if data_entry.mask is not None:
            if not torch.is_tensor(data_entry.mask):
                raise Exception("The mask is not a tensor")
            if data_entry.mask.dtype != torch.bool:
                raise Exception("The mask is not a boolean tensor")
            return data_entry

        inferred_batch_size = data_entry.infer_batch_size()
        inferred_device = data_entry.infer_device()

        if batch_size is not None and inferred_batch_size is not None and batch_size != inferred_batch_size:
            raise ValueError("The provided batch size {} does not match the inferred batch size {}".format(batch_size, inferred_batch_size))
        if device is not None and inferred_device is not None and device != inferred_device:
            raise ValueError("The provided device {} does not match the inferred device {}".format(device, inferred_device))

        if batch_size is None and inferred_batch_size is None:
            raise ValueError("Cannot infer the batch size")
        if device is None and inferred_device is None:
            raise ValueError("Cannot infer the device")

        batch_size_to_use = batch_size if batch_size is not None else inferred_batch_size
        device_to_use = device if device is not None else inferred_device

        new_mask = torch.ones((batch_size_to_use,), device=device_to_use, dtype=torch.bool)
        masked_data_entry = data_entry.shallow_copy()
        masked_data_entry.mask = new_mask
        return masked_data_entry

    def compute_batch_ids_to_preserve(self, batch_size: int, batch_ids: List[int]) -> List[bool]:
        """
        Computes a the of batch ids whose mask information must not be touched
        :param batch_size:
        :param batch_ids:
        :return:
        """
        if batch_ids is None:
            batch_ids = list(range(batch_size))
        batch_ids_to_preserve = list(set(range(batch_size)) - set(batch_ids))

        return batch_ids_to_preserve

    def apply(self, data_entries: DataEntries, batch_ids: List[int]=None) -> DataEntries:
        """
        Applies the masking strategy to the data
        :param data_entries: the data to which to apply masking
        :param batch_ids: the batch element ids to which to apply the masking. If None applies to all batch elements
        :return: the data entries with masking applied
        """
        raise NotImplementedError

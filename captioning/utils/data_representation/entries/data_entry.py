from __future__ import annotations

import collections
import copy
from typing import Any

import numpy as np
import torch


class DataEntry:
    """
    Class representing an entry of data
    """
    def __init__(self, key: str, data: Any, mask: Any=None, type: str=None):
        """
        :param key: Key associated to the data entry
        :param data: Data associated to the data entry
        :param mask: An optional mask associated to the data
        :param type: An optional tag associated to the data entry
        """
        self.key = key
        self.data = data
        self.mask = mask
        self.type = type

    def __repr__(self):
        return f"DataEntry(key={self.key}, data={self.data}, mask={self.mask}, type={self.type})"

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, value):
        if value is not None and not torch.is_tensor(value):
            raise ValueError("The mask must be a tensor")
        if value is not None and value.dtype != torch.bool:
            raise ValueError("The mask must be a boolean tensor")

        self._mask = value

    def infer_device(self):
        """
        Infers the device on which the data is located
        :return: The inferred device or None if the device could not be inferred
        """
        if torch.is_tensor(self.data):
            if torch.is_tensor(self.mask):
                if self.data.device != self.mask.device:
                    raise RuntimeError(f"Data and mask are on different devices: {self.data.device} and {self.mask.device}")
            return self.data.device
        elif torch.is_tensor(self.mask):
            return self.mask.device

        return None

    def infer_batch_size(self) -> int:
        """
        Infers the batch size of the data
        :return: The inferred batch size or None if the batch size could not be inferred
        """
        if torch.is_tensor(self.data) or isinstance(self.data, np.ndarray):
            if torch.is_tensor(self.mask):
                if self.data.shape[0] != self.mask.shape[0]:
                    raise RuntimeError(f"Data and mask have different batch sizes: {self.data.shape[0]} and {self.mask.shape[0]}")
            return self.data.shape[0]
        elif torch.is_tensor(self.mask) or isinstance(self.mask, np.ndarray):
            return self.mask.shape[0]
        # If the data is a sequence, we assume that it is a batch of data.
        # We exclude strings since it is unlikely that in the foreseen applications the length of the string is the batch size
        if isinstance(self.data, collections.Sequence) and not isinstance(self.data, str):
            return len(self.data)

        return None

    def to(self, device=None, dtype=None) -> DataEntry:
        """
        Sets the device and dtype of the data and mask if they are tensors
        :return: A new data entry with the applied changes
        """
        new_entry = self.shallow_copy()

        if torch.is_tensor(new_entry.data):
            new_entry.data = new_entry.data.to(device=device, dtype=dtype)
        if torch.is_tensor(new_entry.mask):
            new_entry.mask = new_entry.mask.to(device=device, dtype=None) # Masks must always be boolean, so we do not alter their dtype

        return new_entry

    def shallow_copy(self) -> DataEntry:
        """
        Returns a shallow copy of the current object
        :return:
        """
        return copy.copy(self)

    def deep_copy(self) -> DataEntry:
        """
        Returns a deep copy of the current object
        :return:
        """
        # Checks if the data or masks are tensor and uses clone instead
        # Necessary to keep tensors attached to the computational grpah
        if torch.is_tensor(self.data):
            new_data = self.data.clone()
        else:
            new_data = copy.deepcopy(self.data)
        if torch.is_tensor(self.mask):
            new_mask = self.mask.clone()
        else:
            new_mask = copy.deepcopy(self.mask)

        # Should be unnecessary since strings are immutable
        new_key = copy.deepcopy(self.key)
        new_type = copy.deepcopy(self.type)

        return DataEntry(new_key, new_data, new_mask, new_type)

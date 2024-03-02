from __future__ import annotations

from typing import Sequence, Union, Set, Callable
import torch


from utils.data_representation.entries.data_entry import DataEntry


class DataEntries:
    """
    Dictionary-like class that represents a collection of data with utility functions for manipulating it
    """
    def __init__(self):
        self.data_entries = {}

    def __repr__(self):
        return f"DataEntries(data_entries={self.data_entries})"

    def infer_device(self, keys_filter: Union[str, Set[str]]=None, types_filter: Union[str, Set[str]]=None):
        """
        Infers the device on which the data is located
        :param keys_filter: Keys to which to allow the device to be inferred. Discards all keys that are not matched by the filter. If None no keys are discarded
        :param types_filter: Entry types to which to allow the device to be inferred. Discards all keys that are not matched by the filter. If None no keys are discarded
        :return: The inferred device or None if the device could not be inferred
        """
        selected_device = None
        for current_entry in self.values(keys_filter=keys_filter, types_filter=types_filter):
            current_device = current_entry.infer_device()
            if current_device is not None:
                # Checks for consistency across the different entries
                if selected_device is not None:
                    if selected_device != current_device:
                        raise ValueError(f"Data entries are located on different devices: {selected_device} and {current_device}")
                selected_device = current_device

        return selected_device

    def infer_batch_size(self, keys_filter: Union[str, Set[str]]=None, types_filter: Union[str, Set[str]]=None) -> int:
        """
        Infers the batch size of the data
        :param keys_filter: Keys to which to allow the batch_size to be inferred. Discards all keys that are not matched by the filter. If None no keys are discarded
        :param types_filter: Entry types to which to allow the batch_size to be inferred. Discards all keys that are not matched by the filter. If None no keys are discarded
        :return: The inferred batch size or None if the batch size could not be inferred
        """
        selected_batch_size = None
        for current_entry in self.values(keys_filter=keys_filter, types_filter=types_filter):
            current_batch_size = current_entry.infer_batch_size()
            if current_batch_size is not None:
                # Checks for consistency across the different entries
                if selected_batch_size is not None:
                    if selected_batch_size != current_batch_size:
                        raise ValueError(f"Data entries have different batch sizes: {selected_batch_size} and {current_batch_size}")
                selected_batch_size = current_batch_size

        return selected_batch_size

    def keys(self, keys_filter: Union[str, Set[str]]=None, types_filter: Union[str, Set[str]]=None) -> Sequence[str]:
        """
        Returns the keys of the entries
        :param keys_filter: Keys to which to allow the dtype change. Discards all keys that are not matched by the filter. If None no keys are discarded
        :param types_filter: Entry types to which to allow the dtype change. Discards all keys that are not matched by the filter. If None no keys are discarded
        :return:
        """
        return list([current_entry.key for current_entry in self.values(keys_filter, types_filter)])

    def values(self, keys_filter: Union[str, Set[str]]=None, types_filter: Union[str, Set[str]]=None) -> Sequence[DataEntry]:
        """
        Returns the values of the entries
        :param keys_filter: Keys to which to allow the dtype change. Discards all keys that are not matched by the filter. If None no keys are discarded
        :param types_filter: Entry types to which to allow the dtype change. Discards all keys that are not matched by the filter. If None no keys are discarded
        :return:
        """
        # Applies default values for filters
        if keys_filter is None:
            keys_filter = set(self.data_entries.keys())
        if isinstance(keys_filter, str):
            keys_filter = {keys_filter}
        if types_filter is None:
            types_filter = set([entry.type for entry in self.data_entries.values()])
        if isinstance(types_filter, str):
            types_filter = {types_filter}

        # Filters the entries
        filtered_entries = []
        for current_entry in self.data_entries.values():
            if current_entry.key in keys_filter and current_entry.type in types_filter:
                filtered_entries.append(current_entry)

        return filtered_entries

    def items(self, keys_filter: Union[str, Set[str]]=None, types_filter: Union[str, Set[str]]=None) -> Sequence[tuple[str, DataEntry]]:
        """
        Returns the items of the entries
        :param keys_filter: Keys to which to allow the dtype change. Discards all keys that are not matched by the filter. If None no keys are discarded
        :param types_filter: Entry types to which to allow the dtype change. Discards all keys that are not matched by the filter. If None no keys are discarded
        :return:
        """
        return [(current_entry.key, current_entry) for current_entry in self.values(keys_filter, types_filter)]

    def add(self, data_entry: DataEntry):
        """
        Adds a data entry to the collection
        :param data_entry:
        :return:
        """
        self.data_entries[data_entry.key] = data_entry

    def __setitem__(self, key: str, value: DataEntry):
        if key != value.key:
            raise ValueError(f"Key {key} does not match the key of the value {value.key}")
        self.add(value)

    def get(self, key: str, default=None) -> DataEntry:
        """
        Returns the data entry associated to the specified key
        :param key: Key to retrieve
        :param default: default value to return if the entry is not present
        :return:
        """
        return self.data_entries.get(key, default)

    def __getitem__(self, key: str) -> DataEntry:
        return self.data_entries[key]

    def remove(self, key: str):
        """
        Removes the data entry associated to the specified key
        :param key: Key to remove
        :return:
        """
        if key in self.data_entries:
            del self.data_entries[key]

    def __delitem__(self, key: str):
        self.remove(key)

    def shallow_copy(self) -> DataEntries:
        """
        Creates a shallow copy of the objects
        :return:
        """
        new_data_entries = DataEntries()
        for current_entry in self.values():
            copied_entry = current_entry.shallow_copy()
            new_data_entries.add(copied_entry)
        return new_data_entries

    def deep_copy(self) -> DataEntries:
        """
        Creates a deep copy of the objects
        :return:
        """
        new_data_entries = DataEntries()
        for current_entry in self.values():
            copied_entry = current_entry.deep_copy()
            new_data_entries.add(copied_entry)
        return new_data_entries

    def apply(self, function: Callable, keys_filter: Union[str, Set[str]]=None, types_filter: Union[str, Set[str]]=None) -> DataEntries:
        """
        Applies a function to the specified entries
        :param function: Function to apply. It must take as input a DataEntry and the current DataEntries object and return a DataEntry
        :param keys_filter: Keys to which to apply the function. Discards all keys that are not matched by the filter. If None no keys are discarded
        :param types_filter: Entry types to which to apply the function. Discards all keys that are not matched by the filter. If None no keys are discarded
        :return: Shallow copy of the current object with the applied changes
        """
        new_data_entries = self.shallow_copy()

        filtered_entries = self.values(keys_filter, types_filter)
        for current_entry in filtered_entries:
            new_entry = function(current_entry.shallow_copy(), self.shallow_copy())  # Makes defensive shallow copies of the objects
            new_data_entries.add(new_entry)  # Substitutes the entries that have been modified by the function

        return new_data_entries

    def to(self, device=None, dtype=None, keys_filter: Union[str, Set[str]]=None, types_filter: Union[str, Set[str]]=None) -> DataEntries:
        """
        Sets the dtype of the specified entries
        :param device: The device to set
        :param dtype: The dtype to set
        :param keys_filter: Keys to which to allow the dtype change. Discards all keys that are not matched by the filter. If None no keys are discarded
        :param types_filter: Entry types to which to allow the dtype change. Discards all keys that are not matched by the filter. If None no keys are discarded
        :return: Shallow copy of the current object with the applied changes
        """
        def dtype_change_function(current_entry: DataEntry, data_entries: DataEntries) -> DataEntry:
            return current_entry.to(device=device, dtype=dtype)

        return self.apply(dtype_change_function, keys_filter, types_filter)

    @staticmethod
    def parallel_apply(function: Callable, all_data_entries: Sequence[DataEntries], keys_filter: Union[str, Set[str]]=None, types_filter: Union[str, Set[str]]=None) -> DataEntries:
        """
        Applies a function taking as input all corresponding data entries from the sequence of entries and returns data entries with the corresponding function output
        :param function: Function to apply. It must take as input a tuple of DataEntry and return a DataEntry
        :param all_data_entries: Sequence of DataEntries objects to which to apply the function
        :param keys_filter: Keys to which to apply the function. Discards all keys that are not matched by the filter. If None no keys are discarded
        :param types_filter: Entry types to which to apply the function. Discards all keys that are not matched by the filter. If None no keys are discarded
        :return A new DataEntries object containing the results of the applied function. Only the filtered keys are present
        """

        new_data_entries = DataEntries()
        filtered_keys = all_data_entries[0].keys(keys_filter, types_filter)
        for current_filtered_key in filtered_keys:
            for current_data_entries in all_data_entries:
                if current_filtered_key not in current_data_entries.keys():
                    raise ValueError(f"Key {current_filtered_key} is not present in all data entries. Make sure that all non-filtered keys are present in all data entries.")
            current_entries = [current_data_entries[current_filtered_key] for current_data_entries in all_data_entries]
            new_entry = function(current_entries)
            new_data_entries.add(new_entry)

        return new_data_entries

    @staticmethod
    def concatenate(all_data_entries: Sequence[DataEntries], dim=0, keys_filter: Union[str, Set[str]]=None, types_filter: Union[str, Set[str]]=None) -> DataEntries:
        """
        Concatenates the specified data entries. Only entries matching the filters are concatenated and present in the output
        :param all_data_entries: Sequence of DataEntries objects to concatenate
        :param dim: Dimension along which to concatenate
        :param keys_filter: Keys to which to apply the function. Discards all keys that are not matched by the filter. If None no keys are discarded
        :param types_filter: Entry types to which to apply the function. Discards all keys that are not matched by the filter. If None no keys are discarded
        """

        def concatenate_impl(current_entries: Sequence[DataEntry]) -> DataEntry:
            all_data = [current_entry.data for current_entry in current_entries]
            all_masks = [current_entry.mask for current_entry in current_entries]

            if all([torch.is_tensor(current_data) for current_data in all_data]):
                all_data = torch.cat(all_data, dim=dim)
            elif all([isinstance(current_data, list)] for current_data in all_data):
                if dim != 0:
                    raise ValueError(f"Concatenation of lists is only supported along the first dimension")
                new_all_data = []
                for current_data in all_data:
                    new_all_data.extend(current_data)
                all_data = new_all_data
            elif all([isinstance(current_data, tuple)] for current_data in all_data):
                if dim != 0:
                    raise ValueError(f"Concatenation of lists is only supported along the first dimension")
                new_all_data = []
                for current_data in all_data:
                    new_all_data.extend(current_data)
                all_data = tuple(new_all_data)
            else:
                raise ValueError(f"Concatenation of non-tensor data is not supported yet")
            
            
            none_mask_found = False
            non_none_mask_found = False
            for current_mask in all_masks:
                if current_mask is None:
                    none_mask_found = True
                else:
                    non_none_mask_found = True
            if none_mask_found and non_none_mask_found:
                raise ValueError(f"Cannot concatenate entries with None and non-None masks")
            if non_none_mask_found:
                all_masks = torch.cat(all_masks, dim=dim)
            else:
                all_masks = None

            new_data_entry = DataEntry(current_entries[0].key, all_data, all_masks, current_entries[0].type)
            return new_data_entry

        result = DataEntries.parallel_apply(concatenate_impl, all_data_entries, keys_filter, types_filter)
        return result

def main():

    import torch
    import numpy as np

    data_entries = DataEntries()
    data_entries.add(DataEntry("key1", "value1", None, "type1"))
    data_entries.add(DataEntry("key2", torch.zeros((2, 4)), None, "type2"))
    data_entries.add(DataEntry("key3", "value1", torch.zeros((2, 4), dtype=torch.bool), "type1"))
    data_entries.add(DataEntry("key4", torch.zeros((2, 4)), torch.zeros((2, 4), dtype=torch.bool), "type1"))
    data_entries.add(DataEntry("key5", torch.zeros((2, 4)), None, "type2"))
    data_entries.add(DataEntry("key6", "value1", torch.zeros((2, 4), dtype=torch.bool), "type1"))
    data_entries.add(DataEntry("key7", torch.zeros((2, 4)), torch.zeros((2, 4), dtype=torch.bool), "type1"))
    data_entries["key8"] = DataEntry("key8", "value1")

    print(data_entries)

    data_entries.add(DataEntry("key1", "value1_modified", torch.zeros((2, 4), dtype=torch.bool), "type1_modified"))

    gpu_entries = data_entries.to(device="cuda:0")
    bool_entries = data_entries.to(dtype=torch.bool)
    type_1_gpu_entries = data_entries.to(device="cuda:0", types_filter="type1")
    key_3_bool_entries = data_entries.to(dtype=torch.bool, keys_filter="key3")

    none_entry = data_entries.get("key9")
    empty_entry = data_entries.get("key9", default=DataEntries())

    deep_copy = data_entries.deep_copy()

    data_entries["key2"].data += 1

    del data_entries["key1"]
    del data_entries["key7"]

    print(data_entries)


    data_entries2 = DataEntries()
    data_entries2.add(DataEntry("key1", "value1", None, "type1"))
    data_entries2.add(DataEntry("key2", ["a"], None, "type2"))

    data_entries3 = DataEntries()
    data_entries3.add(DataEntry("key1", "value1", None, "type1"))
    data_entries3.add(DataEntry("key2", ["b"], None, "type2"))

    concatenated_data_entries = DataEntries.concatenate([data_entries2, data_entries3], dim=0, keys_filter="key2")

    print(concatenated_data_entries)


if __name__ == "__main__":
    main()

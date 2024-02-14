import torch

from utils.data_representation.data_entries import DataEntries
from utils.data_representation.entries.data_entry import DataEntry
from utils.data_representation.gather_scatter.data_entry_gather_scatter import DataEntryGatherScatter
from utils.data_representation.gather_scatter.gather_scatter_utils import gather_tensor, gather_list, gather_element, scatter_tensor

from typing import List, Tuple, Dict, Union, Set

class BatchDataEntryGatherScatter(DataEntryGatherScatter):
    """
    A data entries gather scatter that operates only on the batch dimension, ignoring the temporal one
    """

    def __init__(self):
        pass

    def gather(self, src_data_entry: DataEntry, indexes: List[Tuple[int, List[int]]]) -> DataEntry:
        """
        Gathers the data entry in the specified indexes
        :param src_data_entry: the data entries to gather
        :param indexes: list containing tuples of batch_idx and sequence of ids in the temporal dimension to gather
        :return: the gathered data entries
        """
        
        new_data_entry = src_data_entry.shallow_copy()
        data = new_data_entry.data
        mask = new_data_entry.mask

        new_data = gather_element(data, indexes)
        new_mask = gather_element(mask, indexes)

        new_data_entry.data = new_data
        new_data_entry.mask = new_mask

        return new_data_entry

    def scatter(self, src_data_entry: DataEntry, indexes: List[Tuple[int, List[int]]], dst_data_entry: DataEntry) -> DataEntry:
        """
        Scatters the data entry in the specified indexes
        :param src_data_entry: the data entries to scatter
        :param indexes: list of size (src_batch_size) containing tuples of batch_idx and sequence of ids in the temporal dimension to scatter. There must be an element in the list for each batch element of the source data entries and
                        the number of ids in each sequence must match the length of the sequence in the source data entries. The values in indexes represent which elements of dst_data_entry the current src_data_entries element will be scattered to
        :param dst_data_entry: the data entries to scatter into
        :return: the scattered data entries
        """

        new_data_entry = dst_data_entry.shallow_copy()
        data = new_data_entry.data

        src_data = src_data_entry.data
        if not torch.is_tensor(src_data):
            raise ValueError("Can only scatter tensors, but got the following element to scatter {}".format(src_data))

        # Scatters the tensor
        new_data = scatter_tensor(src_data_entry.data, indexes, data)
        new_data_entry.data = new_data

        return new_data_entry
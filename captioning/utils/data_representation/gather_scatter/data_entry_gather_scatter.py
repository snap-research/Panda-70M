from utils.data_representation.data_entries import DataEntries
from utils.data_representation.entries.data_entry import DataEntry

from typing import List, Tuple, Dict, Union, Set

class DataEntryGatherScatter:

    def __init__(self):
        pass

    def gather(self, src_data_entries: DataEntry, indexes: List[Tuple[int, List[int]]]) -> DataEntry:
        """
        Gathers the data entry in the specified indexes
        :param src_data_entries: the data entries to gather
        :param indexes: list containing tuples of batch_idx and sequence of ids in the temporal dimension to gather
        :return: the gathered data entries
        """
        raise NotImplementedError()

    def scatter(self, src_data_entries: DataEntry, indexes: List[Tuple[int, List[int]]], dst_data_entries: DataEntry):
        """
        Scatters the data entry in the specified indexes
        :param src_data_entries: the data entries to scatter
        :param indexes: list of size (src_batch_size) containing tuples of batch_idx and sequence of ids in the temporal dimension to scatter. There must be an element in the list for each batch element of the source data entries and
                        the number of ids in each sequence must match the length of the sequence in the source data entries. The values in indexes represent which elements of dst_data_entries the current src_data_entries element will be scattered to
        :param dst_data_entries: the data entries to scatter into
        :return: the scattered data entries
        """
        raise NotImplementedError()
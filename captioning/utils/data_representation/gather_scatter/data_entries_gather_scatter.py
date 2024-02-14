import torch

from utils.data_representation.data_entries import DataEntries
from utils.data_representation.entries.data_entry import DataEntry
from utils.data_representation.gather_scatter.data_entry_gather_scatter import DataEntryGatherScatter
from utils.data_representation.gather_scatter.temporal_data_entry_gather_scatter import TemporalDataEntryGatherScatter
from utils.data_representation.gather_scatter.batch_data_entry_gather_scatter import BatchDataEntryGatherScatter

from typing import List, Tuple, Dict, Union, Set

class DataEntriesGatherScatter:

    def __init__(self, gather_scatter_config: Dict):
        
        self.gather_scatter_config = gather_scatter_config
        data_entry_gather_scatterers_config = gather_scatter_config["data_entry_gather_scatterers"]
        
        # Instantiates the gather scatterers for each type of data entry
        self.all_data_entry_gather_scatterers = []
        self.all_data_entry_gather_scatter_filters = []
        for current_data_entry_gather_scatterer_config in data_entry_gather_scatterers_config:
            # Instantiates each gather scatterer
            current_target = current_data_entry_gather_scatterer_config["target"]
            current_gatherer_scatterer = current_target()
            self.all_data_entry_gather_scatterers.append(current_gatherer_scatterer)

            # Retrieves and records the filters for each gather scatterer
            keys_filter = current_data_entry_gather_scatterer_config["keys_filter"]
            types_filter = current_data_entry_gather_scatterer_config["types_filter"]
            self.all_data_entry_gather_scatter_filters.append((keys_filter, types_filter))

    def add_data_entry_gather_scatterer(self, data_entry_gather_scatterer: DataEntryGatherScatter, keys_filter: Union[str, Set[str]]=None, types_filter: Union[str, Set[str]]=None):
        """
        Adds a new gather scatterer to the list of gather scatterers
        :param data_entry_gather_scatterer: the gather scatterer to add
        :param keys_filter: Keys for which to perform the operation.
        :param types_filter: Entry types for which to perform the operation.
        """
        self.all_data_entry_gather_scatterers.append(data_entry_gather_scatterer)
        self.all_data_entry_gather_scatter_filters.append((keys_filter, types_filter))

    def build_key_to_gather_scatter_map(self, data_entries: DataEntries) -> Dict[str, DataEntryGatherScatter]:
        """
        Given data entries, computes a map from keys to gather scatterers to use for that key
        """

        key_to_gather_scatter_map = {}
        for current_data_entry_gather_scatterer, current_data_entry_gather_scatterer_filters in zip(self.all_data_entry_gather_scatterers, self.all_data_entry_gather_scatter_filters):
            keys_filter, types_filter = current_data_entry_gather_scatterer_filters
            
            for key in data_entries.keys(keys_filter=keys_filter, types_filter=types_filter):
                if key in key_to_gather_scatter_map:
                    raise ValueError("Conflict in gather scatter. Key {} is matched by multiple gather scatterers".format(key))
                key_to_gather_scatter_map[key] = current_data_entry_gather_scatterer

        return key_to_gather_scatter_map

    def gather(self, src_data_entries: DataEntries, indexes: List[Tuple[int, List[int]]], keys_filter: Union[str, Set[str]]=None, types_filter: Union[str, Set[str]]=None, include_unmatched_keys=False) -> Tuple[DataEntries, List[str]]:
        """
        Gathers the data entries in the specified indexes
        :param src_data_entries: the data entries to gather
        :param indexes: list containing tuples of batch_idx and sequence of ids in the temporal dimension to gather
        :param keys_filter: Keys for which to perform the operation.
        :param types_filter: Entry types for which to perform the operation.
        :param include_unmatched_keys: If True, keys that are not matched by any gather scatterer are included in the output as is, otherwise they are discarded
        :return: the gathered data entries, keys that went unmatched by the gather operation
        """
        # Makes a defensive copy
        src_data_entries = src_data_entries.shallow_copy()

        key_to_gather_scatter_map = self.build_key_to_gather_scatter_map(src_data_entries)

        result_data_entries = DataEntries()
        for current_key in src_data_entries.keys(keys_filter=keys_filter, types_filter=types_filter):
            current_src_data_entry = src_data_entries[current_key]
            # If no gatherer is matched, continue
            if current_key not in key_to_gather_scatter_map:
                continue

            # Performs gathering
            current_gather_scatterer = key_to_gather_scatter_map[current_key]
            current_result_data_entry = current_gather_scatterer.gather(current_src_data_entry, indexes)
            result_data_entries.add(current_result_data_entry)

        unmatched_keys = []
        # Makes sure all keys in src are copied into dst
        for current_key in src_data_entries.keys():
            if current_key not in result_data_entries.keys():
                unmatched_keys.append(current_key)
                if include_unmatched_keys:
                    result_data_entries.add(src_data_entries[current_key])

        return result_data_entries, unmatched_keys

    def scatter(self, src_data_entries: DataEntries, indexes: List[Tuple[int, List[int]]], dst_data_entries: DataEntries, keys_filter: Union[str, Set[str]]=None, types_filter: Union[str, Set[str]]=None, include_unmatched_keys=False) -> Tuple[DataEntries, List[str]]:
        """
        Scatters the data entries in the specified indexes. All keys in src_data_entries that are not in dst_data_entries are always omitted.
        :param src_data_entries: the data entries to scatter
        :param indexes: list containing tuples of batch_idx and sequence of ids in the temporal dimension to scatter. There must be an element in the list for each batch element of the destination data entries and
                        the number of ids in each sequence must match the length of the sequence in the destination data entries. The values in indexes represent which elements of src_data_entries to take for each
                        element of dst_data_entries
        :param dst_data_entries: the data entries to scatter into
        :param keys_filter: Keys for which to perform the operation.
        :param types_filter: Entry types for which to perform the operation.
        :param include_unmatched_keys: If True, keys in the dst_data_entries that are not matched by any gather scatterer are included in the output as is, otherwise they are discarded. 
                                       All keys in src_data_entries that are not in dst_data_entries are always omitted
        :return: the scattered data entries, keys that went unmatched by the gather operation
        """
        
        # Makes a defensive copy
        dst_data_entries = dst_data_entries.shallow_copy()

        key_to_gather_scatter_map = self.build_key_to_gather_scatter_map(src_data_entries)

        result_data_entries = DataEntries()
        for current_key in dst_data_entries.keys(keys_filter=keys_filter, types_filter=types_filter):
            current_dst_data_entry = dst_data_entries[current_key]
            # If no gatherer is matched, continue
            if current_key not in key_to_gather_scatter_map:
                continue

            if current_key not in src_data_entries.keys():
                raise ValueError("Key {} is in dst_data_entries and has a matching gather scatterer, but is not in src_data_entries. src_keys: {}".format(current_key, list(src_data_entries.keys())))

            # Performs scattering
            current_gather_scatterer = key_to_gather_scatter_map[current_key]
            current_src_data_entry = src_data_entries[current_key]
            current_result_data_entry = current_gather_scatterer.scatter(current_src_data_entry, indexes, current_dst_data_entry)
            result_data_entries.add(current_result_data_entry)

        # Keep all original entries in dst if so is required
        unmatched_keys = []
        for current_key in dst_data_entries.keys():
            if current_key not in result_data_entries.keys():
                unmatched_keys.append(current_key)
                if include_unmatched_keys:
                    result_data_entries.add(dst_data_entries[current_key])
        
        return result_data_entries, unmatched_keys


def main():

    tensor1d = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    tensor2d = torch.arange(0, 32).reshape(8, 4)

    counter2d = torch.ones_like(tensor2d)

    mask1d = torch.ones((8), dtype=torch.bool)
    mask2d = torch.ones((8, 4), dtype=torch.bool)

    list1d = [0, 1, 2, 3, 4, 5, 6, 7]
    list2d = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9 ,10, 11],
        [12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21 ,22, 23],
        [24, 25, 26, 27],
        [28, 29, 30, 31]
    ]
    list_string1d = ["0", "1", "2", "3", "4", "5", "6", "7"]

    gathering_indexes = [(0, [0, 1]), (0, [2, 3]), (3, [0, 3]), (3, [3, 0])]

    entry_tensor1d = DataEntry(key="entry1", data=tensor1d, mask=None, type="type1")
    entry_list1d = DataEntry(key="entry2", data=list1d, mask=mask1d, type="type2")

    entry_tensor2d = DataEntry(key="entry3", data=tensor2d, mask=mask1d, type="type3")
    entry_list2d = DataEntry(key="entry4", data=list2d, mask=mask2d, type="type4")
    entry_to_ignore = DataEntry(key="ignore", data=list2d, mask=mask2d, type="ignore")
    entry_counter = DataEntry(key="counter", data=counter2d, mask=None, type="counter")

    input_data_entries = DataEntries()
    input_data_entries.add(entry_tensor1d)
    input_data_entries.add(entry_list1d)
    input_data_entries.add(entry_tensor2d)
    input_data_entries.add(entry_list2d)
    input_data_entries.add(entry_counter)
    input_data_entries.add(entry_to_ignore)

    gather_scatter_config = {
        "data_entry_gather_scatterers": [
            {
                "target": TemporalDataEntryGatherScatter,
                "keys_filter": ["entry1", "counter"],
                "types_filter": None
            }, {
                "target": TemporalDataEntryGatherScatter,
                "keys_filter": None,
                "types_filter": ["type3"]
            },
            {
                "target": BatchDataEntryGatherScatter,
                "keys_filter": ["entry4"],
                "types_filter": None
            }, {
                "target": BatchDataEntryGatherScatter,
                "keys_filter": None,
                "types_filter": ["type2"]
            }
        ]
    }

    gather_scatter = DataEntriesGatherScatter(gather_scatter_config)
    
    gathered_entries = gather_scatter.gather(input_data_entries, gathering_indexes)

    scattered_entries = gather_scatter.scatter(gathered_entries, gathering_indexes, input_data_entries, keys_filter=["entry1", "entry3", "counter"], include_unmatched_keys=True)

    pass


if __name__ == "__main__":
    main()
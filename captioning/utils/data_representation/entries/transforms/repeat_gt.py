from typing import Dict

from utils.data_representation.batch_adapters.batch_adapter import BatchAdapter
from utils.data_representation.data_entries import DataEntries
from utils.data_representation.entries.data_entry import DataEntry

from torch.nn import functional as F

class RepeatGT:
    """
    Class that takes original data entries and repeat an input field appending 'gt'
    """
    def __init__(self, config: Dict):
        None

    def __call__(self, data_entries: DataEntries) -> DataEntry:
        data_entries = data_entries.shallow_copy()
        
        for current_input_key in list(data_entries.keys(types_filter="input")):
            current_data_entry = data_entries[current_input_key]
            current_input = current_data_entry.data
            data_entries.add(DataEntry(current_input_key + "_gt", current_input, type="gt"))
            
        return data_entries



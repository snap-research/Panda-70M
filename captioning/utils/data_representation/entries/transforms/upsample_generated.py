from typing import Dict

from utils.data_representation.batch_adapters.batch_adapter import BatchAdapter
from utils.data_representation.data_entries import DataEntries
from utils.data_representation.entries.data_entry import DataEntry

from torch.nn import functional as F

class UpsampleGenerated:
    """
    Class that takes original data entries and upsamples input, keeps old input in seprate entry with _lowres suffix.
    """
    def __init__(self, config: Dict):
        scale_factor = config.get("scale_factor", (1, 1, 1))
        if len(scale_factor) == 2:
            self.scale_factor = [1] + scale_factor
        else:
            self.scale_factor = scale_factor

    def __call__(self, data_entries: DataEntries) -> DataEntries:
        data_entries = data_entries.shallow_copy()
        
        for current_input_key in list(data_entries.keys(types_filter="input")):
            current_data_entry = data_entries[current_input_key]
            current_input = current_data_entry.data 
            is_video = (len(current_input.shape) == 5)
            if is_video:
                high_res = current_input.permute(0, 2, 1, 3, 4)
                high_res = F.interpolate(high_res, scale_factor=self.scale_factor, mode='trilinear', align_corners=True)
                high_res = high_res.permute(0, 2, 1, 3, 4)
            else:
                high_res = F.interpolate(current_input, scale_factor=self.scale_factor[1:], mode='bilinear', align_corners=True)

            data_entries.add(DataEntry(current_input_key + "_lowres", current_input, type="lowres_condition"))
            data_entries.add(DataEntry(current_input_key, high_res, type="input"))
            
        return data_entries



from typing import Dict

from utils.data_representation.batch_adapters.batch_adapter import BatchAdapter
from utils.data_representation.data_entries import DataEntries
from utils.data_representation.entries.data_entry import DataEntry

from torch.nn import functional as F

class Downsample:
    """
    Class that takes original data entries and append additional lowres field into it.
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
                low_res = current_input.permute(0, 2, 1, 3, 4)
                low_res = F.interpolate(low_res, scale_factor=self.scale_factor, mode='trilinear', align_corners=True)
                low_res = low_res.permute(0, 2, 1, 3, 4)
            else:
                low_res = F.interpolate(current_input, scale_factor=self.scale_factor[1:], mode='bilinear', align_corners=True)

            data_entries.add(DataEntry(current_input_key + "_lowres", low_res, type="lowres_condition"))
            
        return data_entries



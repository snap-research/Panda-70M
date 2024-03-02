from typing import Dict

from utils.data_representation.data_entries import DataEntries


class Compose:
    """
    Class that compose several transforms 
    """
    def __init__(self, config: Dict):
        self.transforms = []
        for transform_config in config['transforms']:
            transform = transform_config["target"]
            self.transforms.append(transform(transform_config))

    def __call__(self, data_entries: DataEntries) -> DataEntries:            
        for transform in self.transforms:
            data_entries = transform(data_entries)
        return data_entries



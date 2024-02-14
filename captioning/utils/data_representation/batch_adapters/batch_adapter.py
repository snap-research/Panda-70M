from typing import Dict, Any

from utils.data_representation.data_entries import DataEntries


class BatchAdapter:
    """
    Base class representing an modules converting the from the batch data format to the data entries format expected by the network
    Enables decoupling between the dataset output format and the model input format
    """
    def __init__(self, adapter_config: Dict):
        if "transform" in adapter_config:
            transform_config = adapter_config["transform"]
            transform = transform_config["target"]
            self.transform  = transform(transform_config)
        else:
            self.transform = lambda x: x
        self.adapter_config = adapter_config

    def batch_to_data_entries(self, batch: Any) -> DataEntries:
        """
        Converts a batch to the corresponding data entries
        :param batch: The batch to convert
        :return: The batch converted to data entries
        """
        raise NotImplementedError()


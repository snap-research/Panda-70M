from typing import Dict, List
import torch

from utils.data_representation.data_entries import DataEntries
from utils.data_representation.entries.data_entry import DataEntry
from utils.data_representation.masking.masking_strategy import MaskingStrategy


class UnionMaskingStrategy(MaskingStrategy):
    """
    Masking strategy that applies a set of masking strategies to the data
    """
    def __init__(self, masking_config: Dict):
        self.masking_config = masking_config

        # Instantiates the masking strategies
        self.masking_strategies = []
        for masking_strategy_config in self.masking_config["masking_strategies"]:
            target_mask_strategy = masking_strategy_config["target"]
            current_masking_config = target_mask_strategy(masking_strategy_config)
            self.masking_strategies.append(current_masking_config)

    def apply(self, data_entries: DataEntries, batch_ids: List[int]=None) -> DataEntries:
        """
        Applies the masking strategy to the data
        :param data_entries: the data to which to apply masking
        :param batch_ids: the batch element ids to which to apply the masking. If None applies to all batch elements
        :return: the data entries with masking applied
        """
        masked_data_entries = data_entries
        for masking_strategy in self.masking_strategies:
            masked_data_entries = masking_strategy.apply(masked_data_entries, batch_ids)
        return masked_data_entries

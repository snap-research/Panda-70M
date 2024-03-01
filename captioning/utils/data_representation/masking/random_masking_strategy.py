from typing import Dict, List

import numpy as np
import torch

from utils.data_representation.data_entries import DataEntries
from utils.data_representation.entries.data_entry import DataEntry
from utils.data_representation.masking.masking_strategy import MaskingStrategy


class RandomMaskingStrategy(MaskingStrategy):
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

        self.strategy_indexes = np.array(range(len(self.strategies)))
        self.strategy_weights = np.asarray(masking_config["strategy_weights"])
        self.strategy_weights = self.strategy_weights / np.sum(self.strategy_weights)

    def apply(self, data_entries: DataEntries, batch_ids: List[int]=None) -> DataEntries:
        """
        Applies the masking strategy to the data
        :param data_entries: the data to which to apply masking
        :param batch_ids: the batch element ids to which to apply the masking. If None applies to all batch elements
        :return: the data entries with masking applied
        """
        batch_size = data_entries.infer_batch_size()

        if batch_ids is None:
            batch_ids = list(range(batch_size))

        # (len(batch_ids))
        chosen_strategy = np.random.choice(self.strategy_indexes, len(batch_ids), p=self.strategy_weights)

        masked_data_entries = data_entries
        # Applies each strategy to its indexes
        for strategy_idx, strategy in enumerate(self.strategies):
            strategy_batch_ids = [batch_ids[i] for i in range(len(batch_ids)) if chosen_strategy[i] == strategy_idx]
            # Applies the strategy only if it has been selected at least once
            if len(strategy_batch_ids) > 0:
                masked_data_entries = strategy.apply(masked_data_entries, strategy_batch_ids)

from __future__ import annotations

import numpy
import torch
import torch.distributed
import torch.nn as nn
import einops

from utils.tensors.tensor_folder import TensorFolder

class DistributedStatisticsAccumulator:
    """
    Utility class to handle accumulation of statistics across distirbuted processes
    """
    def __init__(self):
        """
        """
        self.accumulator = None
        self.accumulated_samples_count = 0

    def accumulate(self, statistics: torch.Tensor):
        """
        Accumulates statistics about the regions
        :param statistics: (..., features_count) tensor of statistics to accumulate. The ... dimensions are flattened and considered as the number of samples
        """
        statistics, _ = TensorFolder.flatten(statistics, -1)

        samples_count = statistics.shape[0]

        statistics = statistics.sum(dim=0).to(dtype=torch.float64)   # Keeps it in double precision to avoid overflow

        if self.accumulator is None:
            self.accumulator = statistics
        else:
            self.accumulator += statistics

        self.accumulated_samples_count += samples_count

    def compute_statistics(self) -> torch.Tensor:
        """
        Computes the statistics and returns them
        """
        total_samples_count = self.compute_samples_count()

        accumulated_statistics = self.accumulator.clone()
        # Calculate grand totals
        torch.distributed.all_reduce(accumulated_statistics)

        accumulated_statistics = accumulated_statistics / total_samples_count

        return accumulated_statistics

    def compute_samples_count(self) -> int:
        """
        Computes the total number of accumulated samples
        """
        if self.accumulator is None:
            return 0

        current_samples_count = torch.as_tensor(self.accumulated_samples_count, dtype=torch.int64, device=self.accumulator.device)
        # Uses distirbuted to get the current images from all processes
        torch.distributed.all_reduce(current_samples_count)

        total_samples_count = int(current_samples_count.item())

        return total_samples_count



from __future__ import annotations

import numpy
import torch
import torch.distributed
import torch.nn as nn
import einops

class CorrelatedNoiseRegion:

    def __init__(self, h_size: int, w_size: int, t_size: int):
        """
        Instantiates a region with the given height, with and temporal span
        """
        self.h_size = h_size
        self.w_size = w_size
        self.t_size = t_size

    @property
    def size(self):
        """
        Returns the size of the region
        """
        return self.h_size * self.w_size * self.t_size

    def is_independent_from(self, h_dist: int, w_dist: int, t_dist: int) -> bool:
        """
        Checks if the RV at the given distance is independent from the current one
        """
        if h_dist >= self.h_size or w_dist >= self.w_size or t_dist >= self.t_size:
            return True

        return False

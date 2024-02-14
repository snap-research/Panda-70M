"""Base subsampler class"""
from abc import abstractmethod


class Subsampler:
    """Subsamples input and returns in same format (stream dict + metadata)"""

    @abstractmethod
    def __call__(self, streams, metadata):
        raise NotImplementedError("Subsampler should not be called")

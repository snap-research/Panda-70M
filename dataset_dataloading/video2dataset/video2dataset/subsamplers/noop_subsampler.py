"""No operation subsampler"""

from .subsampler import Subsampler


class NoOpSubsampler(Subsampler):
    def __init__(self):
        pass

    def __call__(self, streams, metadata):
        return streams, [metadata], None

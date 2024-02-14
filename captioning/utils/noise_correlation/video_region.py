from __future__ import annotations

import numpy
import torch
import torch.distributed
import torch.nn as nn
import einops

class VideoRegion:

    def __init__(self, h_size: int, w_size: int, t_size: int):
        """
        Instantiates a region with the given height, with and temporal span
        """
        self.h_size = h_size
        self.w_size = w_size
        self.t_size = t_size

        self.first_moment_accumulator = None
        self.second_moment_accumulator = None
        self.accumulated_samples_count = 0

        self.variance = None

    @property
    def size(self):
        """
        Returns the size of the region
        """
        return self.h_size * self.w_size * self.t_size

    def count_independent_rvs_in(self, other: VideoRegion) -> float:
        """
        Counts the number of independent random variables of this region that are present in each of the other region
        :param other: The other region
        :return: The number of independent random variables of this region in the other region
        """
        clipped_h = min(self.h_size, other.h_size)
        clipped_w = min(self.w_size, other.w_size)
        clipped_t = min(self.t_size, other.t_size)

        h_times = other.h_size / clipped_h
        w_times = other.w_size / clipped_w
        t_times = other.t_size / clipped_t

        return h_times * w_times * t_times

    def count_independent_rvs_repetitions_in(self, other: VideoRegion) -> float:
        """
        Counts the number of times each independent random variable of this region in the other region is repeated in the other region
        :param other: The other region
        """
        clipped_h = min(self.h_size, other.h_size)
        clipped_w = min(self.w_size, other.w_size)
        clipped_t = min(self.t_size, other.t_size)

        return clipped_h * clipped_w * clipped_t

    def video_to_regions(self, video: torch.Tensor) -> torch.Tensor:
        """
        Converts a video to regions
        :param video: (batch_size, frames_count, channels, height, width) tensor of videos or (batch_size, channels, height, width) tensor of images
        :return: (batch_size * regions_count, channels, region_size) tensor of regions
        """
        # Converts images to videos
        if video.ndim == 4:
            video = video.unsqueeze(1)

        if video.ndim != 5:
            raise ValueError("The video must have 4 or 5 dimensions")

        video_height = video.shape[3]
        video_width = video.shape[4]
        video_length = video.shape[1]

        if self.h_size > video_height or self.w_size > video_width or self.t_size > video_length:
            raise ValueError("The region size must be smaller or equal to the video size (H: {} W: {} T: {}) (H: {} W: {} T: {})".format(self.h_size, self.w_size, self.t_size, video_height, video_width, video_length))

        if video_height % self.h_size != 0:
            video_height = video_height - (video_height % self.h_size)
        if video_width % self.w_size != 0:
            video_width = video_width - (video_width % self.w_size)
        if video_length % self.t_size != 0:
            video_length = video_length - (video_length % self.t_size)

        height_regions_count = video_height // self.h_size
        width_regions_count = video_width // self.w_size
        time_regions_count = video_length // self.t_size

        # Clips the video so that it is divisible by the region size
        video = video[:, :video_length, :, :video_height, :video_width]

        # Reshapes the video to (batch_size * regions_count, channels, region_size)
        video = einops.rearrange(video, 'b (t t_size) c (h h_size) (w w_size) -> (b h w t) c (t_size h_size w_size)', t=time_regions_count, h=height_regions_count, w=width_regions_count, h_size=self.h_size, w_size=self.w_size, t_size=self.t_size)

        return video

    def compute_mean_differences(self, regions: torch.Tensor) -> torch.Tensor:
        """
        Subtracts the mean of each region from each region
        :param regions: (batch_size * regions_count, channels, region_size) tensor of regions
        """
        return torch.abs(regions - regions.mean(dim=2, keepdim=True))

    def accumulate_moments(self, regions: torch.Tensor):
        """
        Accumulates statistics about the regions
        :param regions: (batch_size * regions_count, channels, region_size) tensor of regions
        """
        regions_count = regions.shape[0]

        # Computes the mean variation over each channel and considers the variation for each region to be the sum of the mean variation over all channels
        estimated_mean_of_variations = regions.mean(dim=-1).sum(dim=1)

        first_moment_delta = estimated_mean_of_variations.sum(dim=0)
        second_moment_delta = (estimated_mean_of_variations ** 2).sum(dim=0)

        if self.first_moment_accumulator is None:
            self.first_moment_accumulator = first_moment_delta.to(dtype=torch.float64)     # Keeps it in double precision to avoid overflow
            self.second_moment_accumulator = second_moment_delta.to(dtype=torch.float64)
        else:
            self.first_moment_accumulator += first_moment_delta
            self.second_moment_accumulator += second_moment_delta

        self.accumulated_samples_count += regions_count

    def compute_statistics(self):
        """
        Computes the statistics of the region and stores them in the instance
        """
        current_regions_count = torch.as_tensor(self.accumulated_samples_count, dtype=torch.int64, device=self.first_moment_accumulator.device)
        # Uses distirbuted to get the current images from all processes
        torch.distributed.all_reduce(current_regions_count)

        total_samples_count = int(current_regions_count.item())

        first_moment = self.first_moment_accumulator.clone()
        second_moment = self.second_moment_accumulator.clone()

        # Calculate grand totals
        torch.distributed.all_reduce(first_moment)
        torch.distributed.all_reduce(second_moment)

        first_moment = first_moment / total_samples_count
        second_moment = second_moment / total_samples_count

        variance = second_moment - (first_moment ** 2)

        self.variance = variance.to(dtype=torch.float32).item()

    def erase_accumulated_moments(self):
        """
        Erases the accumulated statistics
        """
        self.first_moment_accumulator = None
        self.second_moment_accumulator = None
        self.accumulated_samples_count = 0


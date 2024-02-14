from __future__ import annotations

import numpy
import torch
import torch.distributed
import torch.nn as nn
import einops

class CovariancePairSampler:
    """
    Utility class for sampling pairs of points for covariance estimation
    """
    @staticmethod
    def sample_pairs(video: torch.Tensor, h_space: int=None, w_space:int=None, t_space: int=None):
        """
        :param video (batch_size, frames_count, channels, height, width) tensor of the video
        :param h_space: The space between the sampled points in the height dimension. Only one between h_space, w_space and t_space can be specified
        :param w_space: The space between the sampled points in the width dimension. Only one between h_space, w_space and t_space can be specified
        :param t_space: The space between the sampled points in the temporal dimension. Only one between h_space, w_space and t_space can be specified
        :return (batch_size, sampled_points_count, channels, 2) tensor of the sampled points
        """
        if h_space is not None:
            rearranged_video = einops.rearrange(video, 'b f c h w -> b (f w) h c')
            delta = h_space
        elif w_space is not None:
            rearranged_video = einops.rearrange(video, 'b f c h w -> b (f h) w c')
            delta = w_space
        elif t_space is not None:
            rearranged_video = einops.rearrange(video, 'b f c h w -> b (h w) f c')
            delta = t_space
        else:
            raise ValueError('One between h_space, w_space and t_space must be specified')

        # (batch, samples1, samples2, channels)
        first = rearranged_video[:, :, :-delta]
        second = rearranged_video[:, :, delta:]

        first = einops.rearrange(first, 'b s1 s2 c -> b (s1 s2) c')
        second = einops.rearrange(second, 'b s1 s2 c -> b (s1 s2) c')

        packed_pairs = torch.stack([first, second], dim=-1)

        return packed_pairs


from typing import Union, List

import torch
import torch.nn as nn

class TensorSplitter:

    @staticmethod
    def predecessor_successor_split(tensor: torch.Tensor) -> torch.Tensor:
        """
        Splits a tensor into the second dimension predecessors and successors

        :param tensor: (dim1, dim2, ...) tensor
        :return: (dim1, 0:dim2-1, ...), (dim1, 1:dim2, ...) tensor
        """
        predecessor_tensor = tensor[:, :-1]
        successor_tensor = tensor[:, 1:]

        return predecessor_tensor, successor_tensor

    @staticmethod
    def split(tensor: Union[List[torch.Tensor], torch.Tensor], dim: int, factor: int) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Splits the input tensor along a dimension

        :param tensor: The tensor to split. Can be a list of tensor, in which case each is treated independently
        :return: Tuple of factor tensors whose dimension number dim is reduced by factor factor
        """
        if torch.is_tensor(tensor):
            if tensor.size(dim) % factor != 0:
                raise Exception("Tensor dimension is not a multiple of the split factor")

            splits = torch.split(tensor, tensor.size(dim) // factor, dim=dim)
        else:
            splits = [TensorSplitter.split(current_tensor, dim, factor) for current_tensor in tensor]

        return splits

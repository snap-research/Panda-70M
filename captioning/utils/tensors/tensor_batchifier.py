from typing import List

import torch
import torch.nn as nn

class TensorBatchifier:

    @staticmethod
    def batchify(tensor: torch.Tensor, dim: int, batch_size: int) -> List[torch.Tensor]:
        """
        Splits a tensor along the speficied dimension in tensors with size <= batch_size on that dimension
        :param tensor: tensor to batchify
        :param dim: dimension along which to perform splitting
        :param batch_size: maximum number of elements in the specified dimension in each of the produced splits
        :return: the splitted tensors. All tensors except from the last one are guaranteed to have batch_size element in
                 the specified dimension
        """
        all_splits = []
        total_size = tensor.size(dim)
        original_tensor_size = list(tensor.size())

        # Transforms negative indexing into positive
        if dim < 0:
            dim = len(original_tensor_size) + dim

        # Flattens the dimensions after dim so that we can index
        flat_tensor = tensor.reshape(original_tensor_size[:dim + 1] + [-1])
        current_start_index = 0
        while current_start_index < total_size:
            # Computes the bounds of each split
            current_end_index = current_start_index + batch_size
            current_end_index = min(current_end_index, total_size)
            current_split_size = current_end_index - current_start_index

            # Splits
            current_flat_tensor = flat_tensor[..., current_start_index:current_end_index, :]

            # Adds the dimensions after dim that were removed
            current_tensor = current_flat_tensor.reshape(original_tensor_size[:dim] + [current_split_size] + original_tensor_size[dim + 1:])
            all_splits.append(current_tensor)

            current_start_index = current_end_index

        return all_splits

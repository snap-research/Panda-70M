import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.tensors.tensor_folder import TensorFolder


class TensorResizer:

    @staticmethod
    def resize_as(original_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
        """
        Makes the resolution of the first tensor the same as the second
        The number of dimensions in the two tensors must be the same and equal to 4 or 5

        :param original_tensor: (..., height, width) tensor
               target_tensor: (..., target_height, target_width) tensor
        :return: (..., target_height, target_width) first tensor with the height and width dimensions the same as the target
        """
        dimensions = len(original_tensor.size())
        if len(original_tensor.size()) != len(target_tensor.size()):
            raise Exception("Original and target tensor must have the same number of dimensions")
        if dimensions < 4 or dimensions > 5:
            raise Exception(f"Unsupported number of dimensions ({dimensions})")

        # The original dimensions
        original_height = original_tensor.size(-2)
        original_width = original_tensor.size(-1)
        # The target dimensions for the output
        target_height = target_tensor.size(-2)
        target_width = target_tensor.size(-1)

        # Return if the dimension is already correct
        if original_height == target_height and original_width == target_width:
            return original_tensor

        # Flattens the tensor if needed
        sequence_length = None
        if dimensions == 5:
            sequence_length = original_tensor.size(1)
            original_tensor = TensorFolder.flatten(original_tensor)

        # Resizes the tensor
        original_tensor = F.interpolate(original_tensor, (target_height, target_width), mode="bilinear")

        # Folds the tensor if needed
        if dimensions == 5:
            original_tensor = TensorFolder.fold(original_tensor, sequence_length)

        return original_tensor


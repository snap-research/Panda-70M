from typing import Any, Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class TensorBroadcaster:

    @staticmethod
    def add_dimension(tensor: torch.Tensor, size: int, dim: int) -> torch.Tensor:
        """
        :param tensor: the tensor on which to add the dimension
        :param size: size of the new dimension to add
        :param dim: dimension for the new dimension
        :return: The input tensor with the new dimension added
        """
        # Adds 1 dimension
        tensor = tensor.unsqueeze(dim)

        # Computes the number of times each dimension should be repeated
        dimensions_count = len(tensor.size())
        repeat_counts = [1] * dimensions_count
        repeat_counts[dim] = size

        tensor = tensor.repeat(repeat_counts)

        return tensor

    @staticmethod
    def add_dimension_to_dict(dictionary: Dict[Any, Union[Dict, torch.Tensor]], size: int, dim: int) -> torch.Tensor:
        """
        Adds a dimension to each entry in a dictionary. Works recursively if values are dictionaries themselves

        :param tensor: the tensor on which to add the dimension
        :param size: size of the new dimension to add
        :param dim: dimension for the new dimension
        :return: The input tensor with the new dimension added
        """
        new_dict = {}
        for key, value in dictionary.items():
            if isinstance(value, torch.Tensor):
                new_dict[key] = TensorBroadcaster.add_dimension(value, size, dim)
            elif isinstance(value, dict):
                new_dict[key] = TensorBroadcaster.add_dimension_to_dict(value, size, dim)
            else:
                raise Exception("Unsupported type {}".format(type(value)))

        return new_dict

    @staticmethod
    def add_right_dimensions(tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Adds to the right of tensor the dimensions of target that are not present in tensor on its right

        :param (<dim_1_1, <dim_1_2>, ... <dim_1_n>)
        :param (<dim_2_1, <dim_2_2>, ... <dim_2_n>, <target_dims>)
        :return (<dim_1_1, <dim_1_2>, ... <dim_1_n>, <target_dims>)
        """
        dimensions_count = len(tensor.shape)
        target_dimensions = target.shape[dimensions_count:]

        # Expands the tensor to the right
        for _ in range(len(target_dimensions)):
            tensor = tensor.unsqueeze(-1)

        # Creates a new view of the tensor with the target dimension at the right
        tensor = tensor.expand(*([-1] * dimensions_count), *target_dimensions)

        return tensor





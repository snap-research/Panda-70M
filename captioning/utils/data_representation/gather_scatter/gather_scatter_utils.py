from utils.data_representation.data_entries import DataEntries
from utils.data_representation.entries.data_entry import DataEntry

from typing import List, Tuple, Dict, Union, Set, Any

import torch
import numpy as np
import collections.abc as abc

    
def indexes_to_flat_indexes(indexes: List[Tuple[int, List[int]]], temporal_size: int) -> List[int]:
    """
    Transforms batch ids and temporal ids into an id referring to the dimension created by flattening batch size and sequence length together
    :param indexes: list containing tuples of batch_idx and sequence of ids in the temporal dimension to gather
    """

    new_indexes = []
    for batch_idx, sequence in indexes:
        for current_sequence_element in sequence:
            new_indexes.append(batch_idx * temporal_size + current_sequence_element)
    return new_indexes

def check_indexes_range(indexes: List[Tuple[int, List[int]]], batch_size: int, temporal_size: int):
    """
    Checks that the indexes are in the correct range
    :param indexes: list containing tuples of batch_idx and sequence of ids in the temporal dimension to gather
    :param batch_size: the batch size of the data
    :param temporal_size: the temporal size of the data
    """
    
    for batch_idx, sequence in indexes:
        if not (batch_idx >= 0 and batch_idx < batch_size):
            raise ValueError("Batch index {} is out of range for batch size of {}".format(batch_idx, batch_size))
        for current_sequence_element in sequence:
            if not (current_sequence_element >= 0 and current_sequence_element < temporal_size):
                raise ValueError("Sequence element {} is out of range for temporal size of {}".format(current_sequence_element, temporal_size))


def gather_temporal_tensor(src_tensor: torch.Tensor, indexes: List[Tuple[int, List[int]]]) -> torch.Tensor:
    """
    Gathers the temporal tensor in the specified indexes. Also gathers along the temporal dimension.
    :param src_tensor: the tensor to gather (batch_size, sequence_length, ...) or (batch_size) if the tensor has no temporal dimension
    :param indexes: list containing tuples of batch_idx and sequence of ids in the temporal dimension to gather
    :return: the gathered tensor
    """

    if src_tensor is None:
        return None

    batch_size = src_tensor.shape[0]
    new_batch_size = len(indexes)

    # The tensor is 1D, e.g. a mask and has no temporal dimension
    if len(src_tensor.shape) == 1:
        batch_indexes = [index[0] for index in indexes]
        new_tensor = src_tensor[batch_indexes]

    else:
        temporal_size = src_tensor.shape[1]
        # Checks that indexes are in the correct range to avoid undetected indexing errors once flattening
        check_indexes_range(indexes, batch_size, temporal_size)

        flat_indexes = indexes_to_flat_indexes(indexes, temporal_size)

        flat_src_tensor = src_tensor.reshape(-1, *src_tensor.shape[2:])
        flat_new_tensor = flat_src_tensor[flat_indexes]
        new_tensor = flat_new_tensor.reshape(new_batch_size, -1, *src_tensor.shape[2:])

    return new_tensor

def gather_tensor(src_tensor: torch.Tensor, indexes: List[Tuple[int, List[int]]]) -> torch.Tensor:
    """
    Gathers the tensor in the specified batch indexes. Does not take into account dimensions that are not the batch
    :param src_tensor: the tensor to gather (batch_size, ...)
    :param indexes: list containing tuples of batch_idx and sequence of ids in the temporal dimension to gather
    :return: the gathered tensor
    """

    if src_tensor is None:
        return None

    batch_size = src_tensor.shape[0]

    batch_indexes = [index[0] for index in indexes]
    new_tensor = src_tensor[batch_indexes]

    return new_tensor

def gather_temporal_list(src_list: List, indexes: List[Tuple[int, List[int]]]) -> List:
    """
    Gathers the temporal list in the specified indexes. Also gathers along the temporal dimension.
    :param src_list: the list of shape (batch_size, sequence_length, ...) or (batch_size) to gather
    :param indexes: list containing tuples of batch_idx and sequence of ids in the temporal dimension to gather
    :return: the gathered list
    """

    if src_list is None:
        return None

    batch_size = len(src_list)
    new_batch_size = len(indexes)

    # The list is 1D, e.g. a mask and has no temporal dimension
    if not isinstance(src_list[0], abc.Sequence) or isinstance(src_list[0], str): # We probably don't want to index the characters in a sequence
        batch_indexes = [index[0] for index in indexes]
        new_list = [src_list[batch_idx] for batch_idx in batch_indexes]
    else:
        new_list = []
        for current_batch_idx, current_sequence_ids in indexes:
            new_batch_element = []
            for current_sequence_id in current_sequence_ids:
                new_batch_element.append(src_list[current_batch_idx][current_sequence_id])
            new_list.append(new_batch_element)

    return new_list

def gather_list(src_list: List, indexes: List[Tuple[int, List[int]]]) -> List:
    """
    Gathers the list in the specified indexes. Only considers the batch dimension in the list
    :param src_list: the list of shape (batch_size, ...) to gather
    :param indexes: list containing tuples of batch_idx and sequence of ids in the temporal dimension to gather
    :return: the gathered list
    """

    if src_list is None:
        return None

    new_list = []
    for batch_idx, _ in indexes:
        new_list.append(src_list[batch_idx])
    return new_list

def gather_element(src: Any, indexes: List[Tuple[int, List[int]]]) -> Any:
    if src is None:
        return src

    if torch.is_tensor(src):
        result = gather_tensor(src, indexes)
    elif isinstance(src, abc.Sequence):
        result = gather_list(src, indexes)
    else:
        raise Exception("Can not gather data of type {}".format(type(src)))

    return result

def gather_temporal_element(src: Any, indexes: List[Tuple[int, List[int]]]) -> Any:
    if src is None:
        return src

    if torch.is_tensor(src):
        result = gather_temporal_tensor(src, indexes)
    elif isinstance(src, abc.Sequence):
        result = gather_temporal_list(src, indexes)
    else:
        raise Exception("Can not gather data of type {}".format(type(src)))

    return result

def scatter_temporal_tensor(src_tensor: torch.Tensor, indexes: List[Tuple[int, List[int]]], dst_tensor: torch.Tensor) -> torch.Tensor:
    """
    Scatters the tensor in the specified indexes. Also scatters along the temporal dimension.
    :param src_tensor: (src_batch_size, src_sequence_length, ...) of (src_batch_size) tensor to scatter
    :param indexes: list of size (src_batch_size) containing tuples of batch_idx and sequence of ids of length src_sequence_length in the temporal dimension to scatter. There must be an element in the list for each batch element of the source data entries and
                    the number of ids in each sequence must match the length of the sequence in the source data entries. The values in indexes represent the destination of each element o src_data_entries in the dst_data_entries
    :param dst_tensor: (dst_batch_size, dst_sequence_length, ...) tensor to scatter into
    """

    # Scatters only on the batch dimension if the src_tensor has no temporal dimension
    if src_tensor.ndim == 1:
        return scatter_tensor(src_tensor, indexes, dst_tensor)

    # Scatters on the batch and temporal dimension
    dst_batch_size = dst_tensor.shape[0]
    dst_temporal_size = dst_tensor.shape[1]

    # Since torch scatter operates only on a single dimension, flattens batch and temporal dimension into a single dimension
    # (dst_batch_size * dst_sequence_length)
    src_temporal_size = src_tensor.shape[1]
    src_batch_size = src_tensor.shape[0]
    # Checks that indexes are in the correct range to avoid undetected indexing errors once flattening
    check_indexes_range(indexes, dst_batch_size, dst_temporal_size)
    flat_indexes = indexes_to_flat_indexes(indexes, dst_temporal_size)
    flat_indexes_tensor = torch.tensor(flat_indexes, dtype=torch.long, device=src_tensor.device)
    flat_indexes_tensor = flat_indexes_tensor.reshape(*flat_indexes_tensor.shape, *([1] * (len(src_tensor.shape) - 2))).expand(-1, *src_tensor.shape[2:])
    flat_src_tensor = src_tensor.reshape(-1, *src_tensor.shape[2:])
    # (dst_batch_size * dst_sequence_length, ...)
    flat_dst_tensor = dst_tensor.reshape(-1, *dst_tensor.shape[2:])

    # Performs the scatter
    flat_result = flat_dst_tensor.clone()
    flat_result.scatter_add_(0, flat_indexes_tensor, flat_src_tensor)

    # Reshapes
    # (dst_batch_size, dst_sequence_length, ...)
    result = flat_result.reshape(dst_batch_size, -1, *dst_tensor.shape[2:])

    return result

def scatter_tensor(src_tensor: torch.Tensor, indexes: List[Tuple[int, List[int]]], dst_tensor: torch.Tensor) -> torch.Tensor:
    """
    Scatters the tensor in the specified indexes
    :param src_tensor: (src_batch_size, ...) tensor to scatter
    :param indexes: list of size (src_batch_size) containing tuples of batch_idx and sequence of ids in the temporal dimension to scatter. There must be an element in the list for each batch element of the source data entries and
                    the number of ids in each sequence must match the length of the sequence in the source data entries. The values in indexes represent which elements of dst_data_entries the current src_data_entries element will be scattered to
    :param dst_tensor: (dst_batch_size, ...) tensor to scatter into
    :return: the scattered tensor
    """

    batch_indexes = [index[0] for index in indexes]
    batch_indexes_tensor = torch.tensor(batch_indexes, dtype=torch.long, device=src_tensor.device)

    # Makes batch indexes tensor of the same shape as the srctensor
    batch_indexes_tensor = batch_indexes_tensor.reshape(*batch_indexes_tensor.shape, *([1] * (len(src_tensor.shape) - 1))).expand(-1, *src_tensor.shape[1:])

    result = dst_tensor.clone()
    result.scatter_add_(0, batch_indexes_tensor, src_tensor)

    return result

def main():
    tensor1d = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    tensor2d = torch.arange(0, 32).reshape(8, 4)

    list1d = [0, 1, 2, 3, 4, 5, 6, 7]
    list2d = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9 ,10, 11],
        [12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21 ,22, 23],
        [24, 25, 26, 27],
        [28, 29, 30, 31]
    ]
    list_string1d = ["0", "1", "2", "3", "4", "5", "6", "7"]

    # 0, 1   2, 3   12, 15  15, 12
    gathering_indexes = [(0, [0, 1]), (0, [2, 3]), (3, [0, 3]), (3, [3, 0])]

    gathered_tensor1d = gather_temporal_tensor(tensor1d, gathering_indexes)
    gathered_tensor2d = gather_temporal_tensor(tensor2d, gathering_indexes)
    gathered_tensor1d_2 = gather_tensor(tensor1d, gathering_indexes)
    
    gethered_list1d = gather_temporal_list(list1d, gathering_indexes)
    gethered_list2d = gather_temporal_list(list2d, gathering_indexes)
    gethered_list1d_2 = gather_list(list1d, gathering_indexes)
    gethered_list_string1d = gather_temporal_list(list_string1d, gathering_indexes)

    scattered_tensor1d = scatter_temporal_tensor(gathered_tensor1d, gathering_indexes, tensor1d)
    scattered_tensor2d = scatter_temporal_tensor(gathered_tensor2d, gathering_indexes, tensor2d)
    scattered_tensor1d_2 = scatter_tensor(gathered_tensor1d_2, gathering_indexes, tensor1d)
    pass

if __name__ == "__main__":
    main()
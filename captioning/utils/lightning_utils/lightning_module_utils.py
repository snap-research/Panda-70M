import lightning.pytorch as pl
import torch
from utils.distributed.distributed_utils import gather_value_dictionary


def get_lightning_module_lr(module: pl.LightningModule, param_group_idx: int=0):
    """
    Lightning returns incorrect values for learning rates if schedulers are used and triaing is restarted from a checkopint
    https://github.com/Lightning-AI/lightning/issues/12812
    The function circumvents the issue and returns the current learning rate
    :param module:
    :return:
    """
    if module.optimizers(use_pl_optimizer=False):
        current_lr = module.optimizers(use_pl_optimizer=False).param_groups[param_group_idx]["lr"]
        return current_lr
    else:
        return 0.0


def log_dictionary(module: pl.LightningModule, dictionary, prefix="", progress_bar=False, batch_size: int = None, gather_all=False):
    """
    Logs a dictionary to the logger

    :param module: the lightning module
    :param dictionary: the dictionary to log
    :param prefix: prefix to add to the keys
    :param progress_bar: whether to log to the progress bar
    :gather_all: whether to gather all the values from all the ranks before logging them
    """

    if gather_all:
        dictionary = gather_value_dictionary(dictionary)

    # Logs to the progress bar
    for key, value in dictionary.items():
        progress_bar_prefix = prefix
        if progress_bar_prefix:
            progress_bar_prefix += "/"
        module.log(f"{progress_bar_prefix}{key}", value, rank_zero_only=True, prog_bar=progress_bar, logger=False, batch_size=batch_size)
    # Logs to the logger. Manually logs to ensure the step is correct at training resume
    manual_log_dictionary(module, dictionary, step=module.global_step, prefix=prefix)


def manual_log_dictionary(module: pl.LightningModule, dictionary, step: int, prefix="", rank_zero_only=True):
    """
    Manually logs a dictionary to the logger

    :param module: the lightning module
    :param dictionary: the dictionary to log
    :param step: the step to log the dictionary
    :param prefix: prefix to add to the keys
    """
    # Log only from rank 0 if requested
    if rank_zero_only and module.global_rank != 0:
        return

    if prefix:
        prefix += "/"
    module.logger.log_metrics({f"{prefix}{k}": v for k, v in dictionary.items()}, step=step)


def detach_dictionary(dictionary):
    """
    Detaches all the tensors in the dictionary

    :param dictionary: the dictionary for which to detach tensors
    """
    for key, value in dictionary.items():
        if torch.is_tensor(value):
            dictionary[key] = value.detach()
        if isinstance(dictionary[key], list):
            new_iterable = []
            for value in dictionary[key]:
                if torch.is_tensor(value):
                    value = value.detach()
                new_iterable.append(value)
            dictionary[key] = new_iterable
        elif isinstance(value, dict):
            detach_dictionary(value)

import os

import lightning
import torch
import torch.nn as nn
import torch.distributed as dist
from lightning.pytorch.loggers import WandbLogger


class WeightLoggingHook:

    def __init__(self, logger: WandbLogger, log_name: str, log_interval: int):
        """
        Creates hook for logging weight statistics
        :param logger: The logger
        :param log_name: Name to use for the logged statistics
        :param log_interval: Number of steps at which to perform logging
        """
        self.logger = logger
        self.log_name = log_name
        self.log_interval = log_interval
        self.current_step = 0

    def __call__(self, module: nn.Module, input, output):

        # Logs the weights just during training
        if module.training:
            # Logs gradient statistics at regular intervals
            if self.current_step % self.log_interval == 0:
                for name, parameter in module.named_parameters():
                    if parameter.requires_grad:
                        with torch.no_grad(): # Probably not necessary, but use it just to be sure not to extend the graph when operating on the weights
                            current_param_log_name = os.path.join(self.log_name, name)

                            weight_std = torch.std(parameter).item()
                            weight_mean = torch.mean(parameter).item()
                            weight_abs = torch.mean(torch.abs(parameter)).item()
                            weight_norm = torch.norm(parameter).item()
                            log_dict = {
                                f"{current_param_log_name}_std": weight_std,
                                f"{current_param_log_name}_mean": weight_mean,
                                f"{current_param_log_name}_abs": weight_abs,
                                f"{current_param_log_name}_norm": weight_norm
                            }

                            # Logs only from the main process
                            rank = 0
                            if dist.is_initialized():
                                rank = dist.get_rank()
                            if rank == 0:
                                # Logs the gradient assuming WandB
                                self.logger.experiment.log(log_dict, commit=False)  # With commit=False the log gets associated to the current step and will be written together will all other logged information for the current step

            self.current_step += 1

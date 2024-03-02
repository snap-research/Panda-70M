import lightning
import torch
import torch.distributed as dist
from lightning.pytorch.loggers import WandbLogger


class GradLoggingHook:

    def __init__(self, logger: WandbLogger, log_name: str, log_interval: int):
        """
        Creates hook for logging gradient statistics
        :param logger: The logger
        :param log_name: Name to use for the logged statistics
        :param log_interval: Number of steps at which to perform logging
        """
        self.logger = logger
        self.log_name = log_name
        self.log_interval = log_interval
        self.current_step = 0

    def __call__(self, grad: torch.Tensor):

        # Logs gradient statistics at regular intervals
        if self.current_step % self.log_interval == 0:
            grad_std = torch.std(grad).item()
            grad_mean = torch.mean(grad).item()
            grad_abs = torch.mean(torch.abs(grad)).item()
            grad_norm = torch.norm(grad).item()
            log_dict = {
                f"{self.log_name}_std": grad_std,
                f"{self.log_name}_mean": grad_mean,
                f"{self.log_name}_abs": grad_abs,
                f"{self.log_name}_norm": grad_norm
            }

            # Logs only from the main process
            rank = 0
            if dist.is_initialized():
                rank = dist.get_rank()
            if rank == 0:
                # Logs the gradient assuming WandB
                self.logger.experiment.log(log_dict, commit=False)  # With commit=False the log gets associated to the current step and will be written together will all other logged information for the current step

        self.current_step += 1

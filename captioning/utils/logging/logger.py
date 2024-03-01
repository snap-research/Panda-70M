import os
import warnings
from typing import Dict

import wandb
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from torch import nn
import torch.distributed as dist

from utils.logging.hooks.grad_logging_hook import GradLoggingHook
from utils.logging.hooks.weight_logging_hook import WeightLoggingHook


class Logger:
    """
    Helper class for managing the logging. Enables utilities such as:
    - Automatic saving of wandb run identifiers to correctly resume training
    - Application of logging callbacks to the training process. Eg. gradient information
    """
    def __init__(self, config: Dict):
        self.config = config

        # Gets the root directory
        self.checkpoints_directory = config["logging"]["checkpoints_directory"]
        # Filename where to save the id of the run. Needed by loggers such as wandb to correctly resume logging
        self.run_id_filename = os.path.join(self.checkpoints_directory, "run_id.txt")
        self.gradient_log_steps = config["logging"].get("gradient_log_steps", 200)
        self.weight_log_steps = config["logging"].get("weight_log_steps", 200)

        # Retrieves existing wandb id or generates a new one
        if self.id_file_exists():
            run_id = self.get_id_from_file()
        else:
            run_id = wandb.util.generate_id()
            self.save_id_to_file(run_id)

        # Sets the environment variables needed by wandb
        self.set_wandb_environment_variables()
        self.project_name = config["logging"]["project_name"]
        # Instantiates the logger only on the main process
        # If this is not done, wandb will crash the application (https://docs.wandb.ai/guides/track/log/distributed-training)
        self.logger = None
        rank = 0
        if dist.is_initialized():
            rank = dist.get_rank()
        elif "HOSTNAME" in os.environ:
            rank = int(os.environ["HOSTNAME"].split("-")[-1])
        else:
            rank = 0
        if rank == 0:
            self.logger = WandbLogger(name=config["logging"]["run_name"], project=self.project_name, id=run_id, config=config)

    def print(self, *args, **kwargs):
        print(*args, **kwargs)

    def get_logger(self):
        return self.logger

    def set_wandb_environment_variables(self):
        """
        Sets the environment variables that are necessary for wandb to work
        :return:
        """
        wandb_key = self.config["logging"]["wandb_key"]
        os.environ["WANDB_API_KEY"] = wandb_key
        wandb_username = self.config["logging"]["wandb_username"]
        os.environ["WANDB_USERNAME"] = wandb_username
        wandb_base_url = self.config["logging"].get("wandb_base_url", "https://snap.wandb.io")
        os.environ["WANDB_BASE_URL"] = wandb_base_url
        wandb_entity = self.config["logging"].get("wandb_entity", "generative-ai")  # Previously "rutils-users"
        os.environ["WANDB_ENTITY"] = wandb_entity

    def register_grad_hooks(self, model: nn.Module):
        """
        Registers grad logging hooks to the model
        :param model: model for which to register the logging hooks
        :return:
        """
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                current_hook = GradLoggingHook(self.logger, f"gradient_statistics/{name}", self.gradient_log_steps)
                parameter.register_hook(current_hook)

    def register_weight_hooks(self, model: nn.Module):
        """
        Registers weight logging hooks to the model
        :param model: model for which to register the logging hooks.
        The model is required to call the forward method for this to work, so it does not currently work with pytorch lightning modules
        If necessary fix this by implementing it with a pytorch lightning callback
        :return:
        """
        current_hook = WeightLoggingHook(self.logger, f"weight_statistics", self.weight_log_steps)
        model.register_forward_hook(current_hook)

    def id_file_exists(self):
        """
        Checks if the wandb id file exists in the checkpoints directory
        :return:
        """
        return os.path.isfile(self.run_id_filename)

    def save_id_to_file(self, run_id):
        """
        Saves the wandb id to a file in the checkpoints directory
        """
        with open(self.run_id_filename, "w") as f:
            f.write(run_id)

    def get_id_from_file(self):
        """
        Reads the id file and returns the id
        :return: run id, None if file does not exist
        """
        if not self.id_file_exists():
            warnings.warn(f"Run ID file does not exist {self.run_id_filename}")
            return None
        with open(self.run_id_filename, "r") as f:
            return f.readline()

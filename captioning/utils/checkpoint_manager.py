from datetime import timedelta
import glob
import os
import warnings
from typing import List, Tuple, Dict

from lightning.pytorch.callbacks import ModelCheckpoint
import lightning.pytorch as pl


class CheckpointManager:

    def __init__(self, config: Dict):
        self.config = config
        self.checkpoints_directory = self.config["logging"]["checkpoints_directory"]

    def get_checkpoints_by_time(self) -> List[str]:
        """
        return: a list of checkpoints filenames sorted by ascending time
        """
        checkpoints_list = list(sorted(glob.glob(os.path.join(self.checkpoints_directory, "step_*.ckpt"))))
        last_checkpoints_list = list(glob.glob(os.path.join(self.checkpoints_directory, "last*.ckpt")))
        timer_checkpoints_list = list(glob.glob(os.path.join(self.checkpoints_directory, "timer*.ckpt")))
        checkpoints_list.extend(last_checkpoints_list)
        checkpoints_list.extend(timer_checkpoints_list)

        checkpoints_list.sort(key=lambda x: os.path.getmtime(x))

        return checkpoints_list

    def load_latest_checkpoint(self, model, **kwargs) -> Tuple[str, pl.LightningModule]:
        """
        Loads the latest checkpoint from the checkpoints directory. Only loads model weights and does not restore training state
        :param model: The model with loaded weights
        :param kwargs: Constructor parameters with which the model was created
        :return: path to the latest checkpoint for resuming training state, model with loaded weights
        """
        checkpoints_list = self.get_checkpoints_by_time()

        latest_ckeckpoint_path = ""
        if checkpoints_list:
            latest_ckeckpoint_path = checkpoints_list[-1]  # Uses the most recent checkpoint
            print(f"Loading checkpoint: {latest_ckeckpoint_path}")
            model = model.load_from_checkpoint(latest_ckeckpoint_path, **kwargs)
            print(f"Loaded checkpoint: {latest_ckeckpoint_path}")
        else:
            print(f"No checkpoint found in '{self.checkpoints_directory}'")

        return latest_ckeckpoint_path, model

    def create_last_checkpoint_callback(self) -> ModelCheckpoint:
        """
        Creates a callback for end of epoch checkpointing
        :return:
        """
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.checkpoints_directory,
            save_on_train_epoch_end=True,
            filename='last'
        )

        return checkpoint_callback

    def create_periodic_checkpoint_callback(self) -> ModelCheckpoint:
        """
        Creates a callback for periodic checkpointing
        :return:
        """
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.checkpoints_directory,
            every_n_train_steps=self.config["training"]["save_steps"],
            filename='step_{step}',
            save_top_k=-1
        )

        return checkpoint_callback

    def create_time_periodic_checkpoint_callback(self) -> ModelCheckpoint:
        """
        Creates a callback for periodic checkpointing at regular time intervals
        :return:
        """
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.checkpoints_directory,
            train_time_interval=timedelta(minutes=self.config["training"].get("save_minutes", 60)),
            filename='timer',
        )

        return checkpoint_callback

    def create_time_periodic_checkpoint_callback_backup(self) -> ModelCheckpoint:
        """
        Creates a callback for periodic checkpointing at regular time intervals.
        Backup copy in case an error is thrown when saving the other time periodic checkpoint.
        Avoids accidental loss of the time checkpoint in case of corruption during saving of the other.
        :return:
        """
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.checkpoints_directory,
            train_time_interval=timedelta(minutes=self.config["training"].get("save_minutes", 60)),
            filename='timer_bak',
        )

        return checkpoint_callback

    def get_all_checkpoint_callbacks(self) -> List[ModelCheckpoint]:
        """
        Returns a list of all checkpoint callbacks.
        :return:
        """
        checkpoint_callbacks = [
            self.create_last_checkpoint_callback(),
            self.create_periodic_checkpoint_callback(),
            self.create_time_periodic_checkpoint_callback(),
            #self.create_time_periodic_checkpoint_callback_backup()
        ]

        return checkpoint_callbacks

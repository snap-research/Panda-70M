import collections
import json
from typing import Dict

import yaml
import os
from pathlib import Path
import copy
import collections
import collections.abc

from omegaconf import OmegaConf
from utils.configuration.base_configuration import BaseConfiguration

from utils.configuration.yaml_include_loader import IncludeLoader


class Configuration(BaseConfiguration):
    """
    Represents the configuration parameters for running the process
    """
    def __init__(self, path):
        """
        Initializes the configuration with contents from the specified file
        :param path: path to the configuration file in json format
        """
        super().__init__(path)

    def create_directory_structure(self):
        """
        Creates the directory structure needed by the configuration
        Eg. logging/checkpoints/results directories
        :return:
        """
        if "checkpoints_directory" in self.config["logging"]:
            Path(self.config["logging"]["checkpoints_directory"]).mkdir(parents=True, exist_ok=True)

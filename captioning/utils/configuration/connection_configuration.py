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


class ConnectionConfiguration(BaseConfiguration):
    """
    Represents the configuration parameters for connecting to the database
    """
    def __init__(self, path):
        """
        Initializes the configuration with contents from the specified file
        :param path: path to the configuration file in json format
        """
        super().__init__(path)

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

from utils.configuration.yaml_include_loader import IncludeLoader


class BaseConfiguration:
    """
    Represents the configuration parameters for running the process
    """
    def __init__(self, path):
        """
        Initializes the configuration with contents from the specified file
        :param path: path to the configuration file in json format
        """
        with open(path, 'r') as f:
            yaml_object = yaml.load(f, IncludeLoader)

        # Loads the configuration file and converts it to a dictionary
        omegaconf_config = OmegaConf.create(yaml_object, flags={"allow_objects": True}) # Uses the experimental "allow_objects" flag to allow classes and functions to be stored directly in the configuration
        self.config = OmegaConf.to_container(omegaconf_config, resolve=True)

        # Checks the configuration
        self.check_config()

        # Creates the directory structure
        self.create_directory_structure()

    def get_config(self):
        return self.config

    def check_config(self):
        """
        Checks that the configuration is well-formed
        Raises an exception if it is not
        :return:
        """
        pass

    def create_directory_structure(self):
        """
        Creates the directory structure needed by the configuration
        Eg. logging/checkpoints/results directories
        :return:
        """
        pass

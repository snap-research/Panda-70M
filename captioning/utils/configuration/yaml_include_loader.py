import os
import json

import yaml
from typing import Any, IO

from utils.dynamic_import import get_class_by_name


class IncludeLoader(yaml.SafeLoader):
    """
    Class extending the YAML Loader to handle nested documents
    YAML Loader with `!include` constructor.
    From: https://gist.github.com/joshbode/569627ced3076931b02f
    """
    def __init__(self, stream: IO) -> None:
        """
        Initialise Loader
        """
        # Registers the current directory as the root directory
        self.root = os.path.curdir

        super().__init__(stream)


def construct_include(loader: IncludeLoader, node: yaml.Node) -> Any:
    """
    Manages inclusion of the file referenced at node
    """
    filename = os.path.abspath(os.path.join(loader.root, loader.construct_scalar(node)))
    extension = os.path.splitext(filename)[1].lstrip('.')

    with open(filename, 'r') as f:
        if extension in ('yaml', 'yml'):
            return yaml.load(f, IncludeLoader) # Check if nested documents are handled correctly
        elif extension in ('json', ):
            return json.load(f)
        else:
            return ''.join(f.readlines())

def construct_module(loader: IncludeLoader, node: yaml.Node) -> Any:
    """
    Manages inclusion of a referenced function into the file
    """
    function_module_name = loader.construct_scalar(node)
    function = get_class_by_name(function_module_name)
    return function



# Registers the loader
yaml.add_constructor('!include', construct_include, IncludeLoader)
yaml.add_constructor('!module', construct_module, IncludeLoader)


def main():
    """
    Main function
    """
    with open('configs/tests/include_loader_test_files/level1.yaml', 'r') as f:
        config = yaml.load(f, IncludeLoader)

    print(config)


if __name__ == "__main__":
    main()

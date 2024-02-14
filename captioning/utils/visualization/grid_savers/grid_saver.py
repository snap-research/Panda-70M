from typing import List, Union, Tuple, Dict, Any

import math
from PIL import Image

class GridSaver:

    def __init__(self):

        self.height = None
        self.width = None
        self.framerate = None

        self.content = None
        self.is_grid = False
        pass

    def is_cell_content(self, cell_content: Any) -> bool:
        """
        Checks if the given content is cell content of one of the correct types that can be opened by the current type of grid
        """

        raise NotImplemented()

    def open_content(self, content: Any) -> Any:
        """
        Reads the content and returns it in the correct format
        Sets the default values for height, width and framerate if they are not already set
        Eg. Reads it from disk if it is a filename
        """
        raise NotImplemented()

    def save(self, filename: str):
        """
        Saves the grid to the given filename
        """
        raise NotImplemented()

    def add_content(self, content: Union[List, List[List]], force_height: int=None, force_width: int=None, force_framerate: float=None):
        """
        Adds content for saving. Each added element can be a filename or the content itself already opened
        The list can be a list of lists in which case it already defines the structure of the grid to save
        """
        self.height = force_height
        self.width = force_width
        self.framerate = force_framerate

        # Opens the content if needed
        # The List is flat
        if self.is_cell_content(content[0]):
            for current_idx in range(len(content)):
                content[current_idx] = self.open_content(content[current_idx])
            self.is_grid = False
        # The content is already organized in a grid
        else:
            for current_idx in range(len(content)):
                current_row = content[current_idx]
                for current_col_idx in range(len(current_row)):
                    current_row[current_col_idx] = self.open_content(current_row[current_col_idx])
            self.is_grid = True # The content is already a grid

        self.content = content

    def flatten_content(self):
        """
        Makes the content flat instead of a grid
        """
        if self.is_grid:
            self.content = [item for sublist in self.content for item in sublist]
            self.is_grid = False

    def make_grid_by_gridsize(self, max_rows: int=None, max_cols: int=None, max_height: int=None, max_width: int=None):
        """
        Makes the grid by the given grid size constraints. Only one constraint can be given
        :param max_rows: The maximum number of rows
        :param max_cols: The maximum number of columns
        :param max_height: The maximum height of the grid
        :param max_width: The maximum width of the grid
        """
        # Makes sure the content is not already a grid
        self.flatten_content()

        elements_count = len(self.content)
        if max_rows is not None:
            rows = max_rows
            cols = math.ceil(elements_count / rows)
        elif max_cols is not None:
            cols = max_cols
            rows = math.ceil(elements_count / cols)
        elif max_height is not None:
            rows = math.floor(max_height / self.height)
            cols = math.ceil(elements_count / rows)
        elif max_width is not None:
            cols = math.floor(max_width / self.width)
            rows = math.ceil(elements_count / cols)

        # Makes the grid
        new_grid = []
        for current_row in range(rows):
            new_row = []
            for current_col in range(cols):
                curren_content_idx = current_row * cols + current_col
                if curren_content_idx < elements_count:
                    new_row.append(self.content[current_row * cols + current_col])
            new_grid.append(new_row)

        self.content = new_grid
        self.is_grid = True

    def make_grid_by_rowkeys(self, content_keys: List[str]):
        """
        Makes a grid where each row contains the entries for a given key
        :param content_keys: The keys for each content element
        """
        # Makes sure the content is not already a grid
        self.flatten_content()

        if len(content_keys) != len(self.content):
            raise ValueError("The number of content keys {} does not match the number of content elements {}".format(len(content_keys), len(self.content)))

        new_grid = []
        processed_keys = set()
        for current_key in content_keys:
            if current_key in processed_keys:
                continue

            current_row = []
            for current_content, current_content_key in zip(self.content, content_keys):
                if current_content_key == current_key:
                    current_row.append(current_content)

            new_grid.append(current_row)
            processed_keys.add(current_key)

        self.content = new_grid
        self.is_grid = True

    def make_blank_frame(self, height: int, width: int) -> Image.Image:
        """
        Makes a blank frame with the given height, width and framerate
        """
        return Image.new('RGB', (width, height))


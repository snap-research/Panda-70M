from pathlib import Path
from typing import List, Union, Tuple, Dict, Any

import PIL.Image as Image
import os
from utils.image.image_utils import find_image_files

from utils.visualization.grid_savers.grid_saver import GridSaver

class ImageGridSaver(GridSaver):

    def __init__(self):
        pass

    def is_cell_content(self, cell_content: Any) -> bool:
        """
        Checks if the given content is cell content of one of the correct types that can be opened by the current type of grid
        """
        if isinstance(cell_content, str):
            return True
        elif isinstance(cell_content, Image.Image):
            return True

        return False

    def open_content(self, content: Any) -> Any:
        """
        Reads the content and returns it in the correct format
        Sets the default values for height, width and framerate if they are not already set
        Eg. Reads it from disk if it is a filename
        """
        if isinstance(content, str):
            content = Image.open(content)

        if self.height is None:
            self.height = content.size[1]
        if self.width is None:
            self.width = content.size[0]
        # Framerate is not relevant for images

        # Resizes the images
        if self.height != content.size[1] or self.width != content.size[0]:
            content = content.resize((self.width, self.height))

        return content

    def save(self, filename: str):
        """
        Saves the grid to the given filename
        """
        directory = os.path.dirname(filename)
        Path(directory).mkdir(parents=True, exist_ok=True)

        rows = len(self.content)
        # The maximum number of columns
        max_cols = 0
        for current_row in self.content:
            max_cols = max(max_cols, len(current_row))

        # Resolution of the output image
        output_resolution = (self.width * max_cols, self.height * rows)
        grid = self.make_blank_frame(output_resolution[1], output_resolution[0])

        for current_row_idx, current_row in enumerate(self.content):
            for current_col_idx in range(max_cols):
                if current_col_idx < len(current_row):
                    current_cell = current_row[current_col_idx]
                else:
                    current_cell = self.make_blank_frame(self.height, self.width)

                # Pastes the content onto the grid
                grid.paste(current_cell, (current_col_idx * self.width, current_row_idx * self.height))

        # Saves the grid
        grid.save(filename)


def main():
    image_directory = "results/026_kinetics_image_pretrained_009_400k_small_noemabug_40gpus_run_1/step_step=250000.ckpt/class_conditioned_image_quality_seeded_deterministic/deterministic_sampler_edm"
    max_images = 300

    image_files = find_image_files(image_directory)
    image_files = image_files[:max_images]

    image_grid_saver = ImageGridSaver()
    image_grid_saver.add_content(image_files)
    image_grid_saver.make_grid_by_gridsize(max_width=1024)
    image_grid_saver.save("results/image_grid_test.png")
    pass

if __name__ == "__main__":
    main()

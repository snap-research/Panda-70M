from pathlib import Path
from typing import List, Union, Tuple, Dict, Any

import PIL.Image as Image
import os
import collections
import collections.abc
from utils.image.image_utils import find_image_files
from utils.video.video_decoder import VideoDecoder
from utils.video.video_encoder import VideoEncoder
from utils.video.video_utils import find_video_files, get_video_duration, get_video_framerate, get_video_frames_count

from utils.visualization.grid_savers.grid_saver import GridSaver

class VideoGridSaver(GridSaver):

    def __init__(self):
        pass

    def is_cell_content(self, cell_content: Any) -> bool:
        """
        Checks if the given content is cell content of one of the correct types that can be opened by the current type of grid
        """
        if isinstance(cell_content, str):
            return True
        elif isinstance(cell_content, collections.abc.Sequence) and isinstance(cell_content[0], Image.Image):
            return True

        return False

    def open_content(self, content: Any) -> Any:
        """
        Reads the content and returns it in the correct format
        Sets the default values for height, width and framerate if they are not already set
        Eg. Reads it from disk if it is a filename
        """
        if isinstance(content, str):
            if self.framerate is None:
                self.framerate = get_video_framerate(content)
            frames_count = get_video_frames_count(content)

            video_decoder = VideoDecoder(content)
            frame_times = []
            current_time = 0.0
            for current_frame_idx in range(frames_count):
                frame_times.append(current_time)
                current_time += 1.0 / self.framerate
            decoded_frames = video_decoder.decode_frames_at_times(frame_times)
            content = decoded_frames

        if self.height is None:
            self.height = content[0].size[1]
        if self.width is None:
            self.width = content[0].size[0]
        # Framerate is not relevant for images

        resized_content = []
        for current_frame in content:
            # Resizes the images
            if self.height != current_frame.size[1] or self.width != current_frame.size[0]:
                current_frame = current_frame.resize((self.width, self.height))
            resized_content.append(current_frame)

        return resized_content

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
        # Gets the maximum number of frames
        max_frames = 0
        for current_row in self.content:
            for current_cell in current_row:
                max_frames = max(max_frames, len(current_cell))

        # Resolution of the output image
        output_resolution = (self.width * max_cols, self.height * rows)

        all_frames = []
        # Makes each frame of the video
        for current_frame_idx in range(max_frames):

            grid = self.make_blank_frame(output_resolution[1], output_resolution[0])

            for current_row_idx, current_row in enumerate(self.content):
                for current_col_idx in range(max_cols):
                    # Add a frame only if that column element is present and the frame is present
                    if current_col_idx < len(current_row) and current_frame_idx < len(current_row[current_col_idx]):
                        current_frame = current_row[current_col_idx][current_frame_idx]
                    else:
                        current_frame = self.make_blank_frame(self.height, self.width)

                    # Pastes the content onto the grid
                    grid.paste(current_frame, (current_col_idx * self.width, current_row_idx * self.height))

            all_frames.append(grid)

        # Saves the video
        video_encoder = VideoEncoder()
        encoding_parameters = {
            "framerate": self.framerate,
        }
        video_encoder.encode_pil_video(all_frames, filename, encoding_parameters)


def main():
    image_directory = "results/026_kinetics_image_pretrained_009_400k_small_noemabug_40gpus_run_1/step_step=250000.ckpt/class_conditioned_video_quality_seeded_deterministic/deterministic_sampler_edm"
    max_images = 300

    image_files = find_video_files(image_directory)
    image_files = image_files[:max_images]

    image_grid_saver = VideoGridSaver()
    image_grid_saver.add_content(image_files)
    image_grid_saver.make_grid_by_gridsize(max_width=1024)
    image_grid_saver.save("results/video_grid_test.mp4")
    pass

if __name__ == "__main__":
    main()

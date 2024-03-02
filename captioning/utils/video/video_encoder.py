import os
import subprocess
from typing import List, Dict, Union

import numpy as np
import torch
from PIL import Image

from utils.video.video_utils import get_video_framerate, get_video_duration, video_encoding_configuration_by_name, \
    get_video_frames_count, get_video_dimensions


class VideoEncoder:
    """
    Utility class to manage the video encoding process
    """
    def __init__(self, ffmpeg_command: str=None):

        if ffmpeg_command is None:
            ffmpeg_command = "/usr/bin/ffmpeg"  # Avoid by default using the one packed in anaconda

        self.ffmpeg_command = ffmpeg_command

    def add_codec_options(self, command: List, encoding_parameters: dict):
        """
        Adds the codec options to the ffmpeg command line
        :param command:
        :param encoding_parameters:
        :return:
        """
        codec = encoding_parameters["codec"]
        gop_size = None
        if "gop_size" in encoding_parameters:
            gop_size = encoding_parameters["gop_size"]
        crf = None
        if "crf" in encoding_parameters:
            crf = encoding_parameters["crf"]
        if "framerate" in encoding_parameters:
            framerate = encoding_parameters["framerate"]

        width = -1
        if "width" in encoding_parameters:
            width = encoding_parameters["width"]
        height = -1
        if "height" in encoding_parameters:
            height = encoding_parameters["height"]

        if codec == "h264":
            command.extend([
                "-vcodec",
                "libx264",
                "-pix_fmt",
                "yuv420p",
            ])
            if crf is not None:
                command.extend([
                    "-crf",
                    str(crf),
                ])
            if gop_size is not None:
                command.extend([
                    "-g",
                    str(gop_size),
                ])

            if "preset" in encoding_parameters:
                command.extend([
                    "-preset",
                    encoding_parameters["preset"]
                ])

            if "tune" in encoding_parameters:
                command.extend(["-tune", encoding_parameters["tune"]])

        elif codec == "h265":
            command.extend([
                "-vcodec",
                "libx265",
                "-pix_fmt",
                "yuv420p",
            ])
            if crf is not None:
                command.extend([
                    "-crf",
                    str(crf),
                ])
            if gop_size is not None:
                command.extend([
                    "-g",
                    str(gop_size),
                ])

            if "preset" in encoding_parameters:
                command.extend([
                    "-preset",
                    encoding_parameters["preset"]
                ])

            if "tune" in encoding_parameters:
                command.extend(["-tune", encoding_parameters["tune"]])

        elif codec == "vp9":
            command.extend([
                "-vcodec",
                "libvpx-vp9",
                "-pix_fmt",
                "yuv420p",
            ])
            if crf is not None:
                command.extend([
                   "-crf",
                   str(crf),
                   "-b:v", "0",
                ])
            if gop_size is not None:
                command.extend([
                   "-g",
                   str(gop_size),
                ])

            if "tile_columns" in encoding_parameters:
                command.extend(["-tile-columns", str(encoding_parameters["tile_columns"])])
            if "row_mt" in encoding_parameters:
                command.extend(["-row-mt", str(encoding_parameters["row_mt"])])

            # Sets up audio coding using Opus
            command.extend(["-c:a", "libopus"])

        elif codec == "ffv1":
            command.extend([
                "-vcodec",
                "ffv1",
                "-level",
                "1",
                "-coder",
                "1",
            ])
            if gop_size is not None:
                command.extend([
                    "-g",
                    str(gop_size)
                ])
        else:
            raise Exception("Unknown codec {}".format(codec))

        if "framerate" in encoding_parameters:
            command.extend([
                "-r",
                str(framerate)
            ])

        # Adds resizing if needed
        if width != -1 or height != -1:
            command.extend(["-vf", "scale={}:{}".format(width, height)])
        elif "max_size" in encoding_parameters:
            max_size = encoding_parameters["max_size"]
            command.extend(["-vf", "scale=if(gte(iw\,ih)\,min({}\,iw)\,-2):if(lt(iw\,ih)\,min({}\,ih)\,-2)".format(max_size, max_size)])

    def encode_file(self, input_filename: str, output_filename: str, encoding_parameters: Dict, split_max_duration: float=None) -> List[str]:
        """
        Re-Encodes a video file using the provided parameters

        :param input_filename: file to re-encode
        :param output_filename: name of the output file to produce
        :param encoding_parameters: dictionary with the encoding parameters
        :param split_max_duration: if not None, encode the video as multiple videos.
                                   Each video has a duration that does not exceed split_max_duration and the output name is modified to include its split number.
        :return: List with the created filenames
        """
        if split_max_duration is None or split_max_duration == 0.0:
            # Builds the command
            command = [self.ffmpeg_command,
                       "-y",
                       "-i", input_filename]

            self.add_codec_options(command, encoding_parameters)
            command.append(output_filename)
            # Runs the command
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            return [output_filename]
        else:

            video_framerate = get_video_framerate(input_filename)
            video_duration = get_video_duration(input_filename)
            # Computes the split_max_duration so that it is an exact multiple of the video framerate.
            # This ensures ffmpeg does not add a trailing frame in case a fractional frame duration needs
            # to be covered at the end of the video, making it exceed the maximum split length
            max_frames = int(video_framerate * split_max_duration)
            frame_corrected_split_max_duration = max_frames / video_framerate

            current_split_start_time = 0
            current_split_idx = 0
            all_split_filenames = []
            while current_split_start_time < video_duration:
                # Builds the command
                command = [self.ffmpeg_command,
                           "-y",
                           '-ss',
                           f'{current_split_start_time}',
                           '-t',
                           f'{frame_corrected_split_max_duration}',
                           "-i", input_filename]

                # Adds the codec options
                self.add_codec_options(command, encoding_parameters)

                # Adds the vsync vfr option to avoid first frame duplication
                command.extend(["-vsync", "vfr"])

                current_split_filename = os.path.splitext(output_filename)[0] + f"_split_{current_split_idx:05d}" + os.path.splitext(output_filename)[1]
                command.append(current_split_filename)
                # Runs the command
                subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                current_split_idx += 1
                current_split_start_time += frame_corrected_split_max_duration
                all_split_filenames.append(current_split_filename)

            return all_split_filenames

    def encode_torch_video(self, frames: torch.Tensor, output_filename: str, encoding_parameters: Dict):
        """
        Encodes a video in torch tensor format

        :param frames: (frames_count, 3, height, width) tensor with video frames with values in [0, 1]
        :param output_filename: name for the output video
        :param encoding_parameters: dictionary with the encoding parameters
        :return:
        """
        # Converts torch frames to numpy (frames_count, height, width, 3)
        converted_frames = (frames * 255).to(torch.uint8).permute((0, 2, 3, 1))
        numpy_frames = converted_frames.detach().cpu().numpy()

        self.encode_numpy_video(numpy_frames, output_filename, encoding_parameters)

    def encode_pil_video(self, frames: List[Image.Image], output_filename: str, encoding_parameters: Dict):
        """
        Encodes a video in PIL format

        :param frames: list of PIL images to save
        :param output_filename: name for the output video
        :param encoding_parameters: dictionary with the encoding parameters
        :return:
        """
        numpy_frames = [np.array(image) for image in frames]
        numpy_frames = np.stack(numpy_frames, axis=0)

        self.encode_numpy_video(numpy_frames, output_filename, encoding_parameters)

    def encode_numpy_video(self, frames: np.ndarray, output_filename: str, encoding_parameters: Dict):
        """
        Encodes a video in numpy format

        :param frames: (frames_count, height, width, 3) uint8 array with the frames to save
        :param output_filename: name for the output video
        :param encoding_parameters: dictionary with the encoding parameters
        :return:
        """
        if "framerate" not in encoding_parameters:
            raise Exception("framerate must be specified for encoding")

        command = [self.ffmpeg_command,
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', '{}x{}'.format(frames.shape[-2], frames.shape[-3]),
            '-pix_fmt', 'rgb24',
            '-r', str(encoding_parameters["framerate"]),  # framerate
            '-i', '-',
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            '-an',  # do not expect audio
            output_filename]

        pipe = subprocess.Popen(command, stdin=subprocess.PIPE)
        pipe.communicate(input=frames.tobytes())  # Use communicate as a better alternative than stdin.write
        #pipe.stdin.write(frames.tostring())
        pipe.stdin.close()


def main():
    file_to_encode = "video.mp4"
    output_file = "video_reencoded.mp4"

    configuration_name = "h265_crf_23_h576"
    configuration = video_encoding_configuration_by_name(configuration_name)

    split_max_duration = 1.01

    video_encoder = VideoEncoder()
    encoded_filenames = video_encoder.encode_file(file_to_encode, output_file, configuration, split_max_duration)
    print("Encoded files: {}".format(encoded_filenames))

if __name__ == "__main__":
    main()

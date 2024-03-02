import glob
import os
import subprocess
from typing import List, Dict


def frame_number_to_timestamp(video_framerate: float, frame_number: int) -> float:
    """
    Converts a frame number to a timestamp
    :param video_framerate: framerate of the video
    :param frame_number: frame number to convert
    :return: timestamp corresponding to the frame number
    """
    return frame_number / video_framerate


def frame_numbers_to_timestamps(video_framerate: float, frame_numbers: List[int]) -> List[float]:
    """
    Converts a list of frame numbers to a list of timestamps
    :param video_framerate: framerate of the video
    :param frame_numbers: list of frame numbers to convert
    :return: list of timestamps corresponding to the frame numbers
    """
    return [frame_number / video_framerate for frame_number in frame_numbers]


def timestamp_to_frame_number(video_framerate: float, timestamp: float) -> int:
    """
    Converts a timestamp to a frame number
    :param video_framerate: framerate of the video
    :param timestamp: timestamp to convert
    :return: frame number corresponding to the timestamp
    """
    return round(timestamp * video_framerate)


def timestamps_to_frame_numbers(video_framerate: float, timestamps: List[float]) -> List[int]:
    """
    Converts a list of timestamps to a list of frame numbers
    :param video_framerate: framerate of the video
    :param timestamps: list of timestamps to convert
    :return: list of frame numbers corresponding to the timestamps
    """
    return [round(timestamp * video_framerate) for timestamp in timestamps]


class maybe_open_file:
    """
    Context manager that manages opening files when the input filename may already be a file handle.
    """
    def __init__(self, filename, *args, **kwargs):
        self.close_on_exit = kwargs.pop('close_on_exit', False)
        if isinstance(filename, str):
            self.file_handle = open(filename, *args, **kwargs)
            self.close_on_exit = True
        else:
            # The argument is already an optn file handle
            self.file_handle = filename

    def __enter__(self):
        return self.file_handle

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.close_on_exit:
            self.file_handle.close()

        return False


def get_video_duration(path: str, ffprobe_command: str="/usr/bin/ffprobe") -> float:
    """
    Returns the duration in seconds of the specified video
    :param path: video file of which to compute the duration
    :return:
    """
    framerate = get_video_framerate(path, ffprobe_command)
    frames = get_video_frames_count(path, ffprobe_command)
    return frames / framerate


def get_approximated_video_duration(path: str, ffprobe_command: str= "/usr/bin/ffprobe") -> float:
    """
    Returns the approximated duration in seconds of the specified video.
    For a precise but slow computation see get_video_duration
    :param path: video file of which to compute the duration
    :return:
    """
    pipe = subprocess.Popen([ffprobe_command, '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', path], stdout=subprocess.PIPE).stdout
    output = pipe.read()
    duration = float(output)
    return duration


def get_video_frames_count(path: str, ffprobe_command: str="/usr/bin/ffprobe"):
    """
    Returns the number of frames in the video
    :param path: path of the video
    :return: number of frames in the video
    """
    pipe = subprocess.Popen([ffprobe_command, '-v', 'error', '-select_streams', 'v:0', '-count_frames', '-show_entries', 'stream=nb_read_frames', '-of', 'csv=p=0', path], stdout=subprocess.PIPE).stdout
    output = pipe.read()
    frames = int(output)

    return frames


def get_video_framerate(path: str, ffprobe_command: str= "/usr/bin/ffprobe"):
    """
    Returns the fps of the specified video
    :param path: video file of which to compute the duration
    :return:
    """
    pipe = subprocess.Popen([ffprobe_command, '-v', 'error', '-select_streams', 'v', '-of', 'default=noprint_wrappers=1:nokey=1', '-show_entries', 'stream=r_frame_rate', path], stdout=subprocess.PIPE).stdout
    output = pipe.read()

    # Results are of the form eg. 25/1
    packed_output = output.decode("utf-8").split("/")
    if len(packed_output) != 2:
        raise ValueError("Unexpected output from ffprobe: '{}'. Video at path '{}' appears to be corrupted.".format(packed_output, path))
    numerator, denominator = packed_output

    framerate = float(numerator) / float(denominator)
    return framerate

def get_average_bitrate(path: str, ffprobe_command: str= "/usr/bin/ffprobe"):
    """
    Returns the estimated average bitrate in kbps of the video based on file size and duration
    :param path: video file of which to compute the bitrate
    :return:
    """
    duration = get_video_duration(path, ffprobe_command)
    size = os.path.getsize(path)

    return size / duration / 1024 * 8


def get_video_dimensions(path: str, ffprobe_command: str= "/usr/bin/ffprobe"):
    """
    Returns the dimensions of the specified video
    :param path: video file of which to compute the duration
    :return: width, height of the video
    """
    pipe = subprocess.Popen([ffprobe_command, '-v', 'error', '-select_streams', 'v', '-of', 'default=noprint_wrappers=1:nokey=1', '-show_entries', 'stream=width,height', path], stdout=subprocess.PIPE).stdout
    output = pipe.read()

    # Results are of the form eg. 1280\n720\n
    width, height = output.decode("utf-8").split("\n")[:2]

    return int(width), int(height)


def get_video_information(output_filename: str, ffprobe_command: str= "/usr/bin/ffprobe") -> Dict:
    """
    Computes a range of information for a given video
    :param output_filename: name of the encoded video
    :return: dictionary with the information
    """
    result = {
        "size": os.path.getsize(output_filename),
        "framerate": get_video_framerate(output_filename, ffprobe_command),
        "frames_count": get_video_frames_count(output_filename, ffprobe_command),
    }

    width, height = get_video_dimensions(output_filename, ffprobe_command)
    result["duration"] = result["frames_count"] / result["framerate"]
    result["width"] = width
    result["height"] = height
    result["average_bitrate"] = result["size"] * 8 / result["duration"] / 1024

    return result


def video_encoding_configuration_by_name(configuration_name: str):
    if configuration_name == "vp9_crf_48":
        configuration = {
            "codec": "vp9",
            "crf": 48,
            "gop_size": 5,
            "row_mt": 1,
        }
    elif configuration_name == "vp9_crf_48_h576":
        configuration = {
            "codec": "vp9",
            "crf": 48,
            "gop_size": 5,
            "row_mt": 1,
            "height": 576,
            "width": 1024,
        }
    elif configuration_name == "vp9_crf_43_h576":
        configuration = {
            "codec": "vp9",
            "crf": 43,
            "gop_size": 5,
            "row_mt": 1,
            "height": 576,
            "width": 1024,
        }
    elif configuration_name == "vp9_crf_41_h576":
        configuration = {
            "codec": "vp9",
            "crf": 41,
            "gop_size": 5,
            "row_mt": 1,
            "height": 576,
            "width": 1024,
        }
    elif configuration_name == "h265_crf_25_h576":
        configuration = {
            "codec": "h265",
            "crf": 25,
            "gop_size": 5,
            "tune": "fastdecode",
            "height": 576,
            "width": 1024,
        }
    elif configuration_name == "h265_crf_23_h576":
        configuration = {
            "codec": "h265",
            "crf": 23,
            "gop_size": 5,
            "tune": "fastdecode",
            "height": 576,
            "width": 1024,
        }
    elif configuration_name == "h265_crf_23_maxs1024":
        configuration = {
            "codec": "h265",
            "crf": 23,
            "gop_size": 5,
            "tune": "fastdecode",
            "max_size": 1024,
        }
    elif configuration_name == "h264_crf_17":
        configuration = {
            "codec": "h264",
            "crf": 17,
            "gop_size": 5,
        }
    elif configuration_name == "h264_crf_17_maxs1024":
        configuration = {
            "codec": "h264",
            "crf": 17,
            "gop_size": 5,
            "max_size": 1024,
        }
    elif configuration_name == "h264_crf_20_maxs1024":
        configuration = {
            "codec": "h264",
            "crf": 20,
            "gop_size": 5,
            "max_size": 1024,
        }
    else:
        raise Exception("Unknown configuration: " + configuration_name)

    return configuration

def find_video_files_recursive(video_directory: str, extensions=["mp4", "mkv", "webm", "avi", "webp"]) -> List[str]:
    """
    Gets all the videos in a directory recursively
    :param video_directory:
    :param extensions:
    :return:
    """
    video_files = glob.glob(os.path.join(video_directory, "*"))

    # Filters the files
    selected_files = []
    for video_file in video_files:
        if is_video_file(video_file, extensions):
            selected_files.append(video_file)
        elif os.path.isdir(video_file):
            recursive_files = find_video_files_recursive(video_file, extensions)
            selected_files.extend(recursive_files)

    return selected_files

def find_video_files(video_directory: str, extensions=["mp4", "mkv", "webm", "avi", "webp"]) -> List[str]:
    """
    Gets all the videos in a directory
    :param video_directory:
    :param extensions:
    :return:
    """
    video_files = glob.glob(os.path.join(video_directory, "*"))

    # Filters the files
    selected_files = []
    for video_file in video_files:
        if is_video_file(video_file, extensions):
            selected_files.append(video_file)

    return selected_files

def is_video_file(filename: str, extensions=["mp4", "mkv", "webm", "avi", "webp"]) -> bool:
    """
    Checks if the file is a video file
    :param filename:
    :param extensions:
    :return:
    """
    for extension in extensions:
        if filename.endswith("." + extension):
            return True

    return False


def video_extension_by_codec(codec: str):
    if codec == "vp9":
        return "webm"
    elif codec == "ffv1":
        return "mkv"
    elif codec == "h264":
        return "mp4"
    elif codec == "h265":
        return "mp4"
    else:
        raise Exception("Unknown codec: " + codec)

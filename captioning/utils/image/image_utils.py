import glob
import os
from typing import List


def image_encoding_configuration_by_name(configuration_name: str):
    if configuration_name == "jpeg_quality_90":
        configuration = {
            "codec": "jpeg",
            "quality": 90,
        }
    elif configuration_name == "jpeg_quality_95":
        configuration = {
            "codec": "jpeg",
            "quality": 95,
        }
    else:
        raise Exception("Unknown configuration: " + configuration_name)

    return configuration


def find_image_files(image_directory: str, extensions=["jpg", "jpeg", "webp", "png", "tiff"]) -> List[str]:
    """
    Gets all the videos in a directory
    :param image_directory:
    :param extensions:
    :return:
    """
    image_files = glob.glob(os.path.join(image_directory, "*"))

    # Filters the files
    selected_files = []
    for image_file in image_files:
        if is_image_file(image_file, extensions):
            selected_files.append(image_file)

    return selected_files

def is_image_file(filename: str, extensions=["jpg", "jpeg", "webp", "png", "tiff"]) -> bool:
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

def image_extension_by_codec(codec: str):
    if codec == "jpeg":
        return "jpg"
    elif codec == "png":
        return "png"
    else:
        raise Exception("Unknown codec: " + codec)

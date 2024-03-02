import os
import subprocess
from typing import List, Dict, Union

import numpy as np
import torch
from PIL import Image


class ImageEncoder:
    """
    Utility class to manage the image encoding process
    """
    def __init__(self):
        pass

    def compute_pil_encoding_parameters(self, encoding_parameters: Dict):
        """
        Computes the encoding parameters for the PIL library

        :param encoding_parameters: dictionary with the encoding parameters
        :return: dictionary with the encoding parameters for the PIL library
        """
        pil_encoding_parameters = {}

        if "quality" in encoding_parameters:
            pil_encoding_parameters["quality"] = encoding_parameters["quality"]

        return pil_encoding_parameters

    def encode_file(self, input_filename: str, output_filename: str, encoding_parameters: Dict) -> str:
        """
        Re-Encodes a video file using the provided parameters

        :param input_filename: file to re-encode
        :param output_filename: name of the output file to produce
        :param encoding_parameters: dictionary with the encoding parameters
        :return: The created filename
        """
        pil_encoding_parameters = self.compute_pil_encoding_parameters(encoding_parameters)
        pil_image = Image.open(input_filename)

        # Checks whether the image needs to be resized
        width = -1
        if "width" in encoding_parameters:
            width = encoding_parameters["width"]
        height = -1
        if "height" in encoding_parameters:
            height = encoding_parameters["height"]
        if width != -1 or height != -1:
            # Resizes maintaining the aspect ratio
            if width == -1:
                width = int(height * pil_image.width / pil_image.height)
            if height == -1:
                height = int(width * pil_image.height / pil_image.width)
            pil_image = pil_image.resize((width, height), Resampling.LANCZOS)

        # Encodes the image with the given parameters
        pil_image.save(output_filename, **pil_encoding_parameters)

        return output_filename

def main():
    pass


if __name__ == "__main__":
    main()

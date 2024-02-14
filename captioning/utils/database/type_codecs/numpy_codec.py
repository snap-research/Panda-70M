import io
import pickle

import numpy as np
from bson.binary import Binary, USER_DEFINED_SUBTYPE
from bson.codec_options import TypeDecoder


class NumpyEncoderDecoder:

    magic_numbers = (b'NPNP', b'NPBI')

    def encode_shape(self, shape: np.ndarray):
        """
        Encodes the shape of a numpy array as a binary string
        :param shape:
        :return:
        """
        shape_length = len(shape)
        shape_binary = shape_length.to_bytes(4, byteorder='big')
        shape_elements_binary = [int(element).to_bytes(4, byteorder='big') for element in shape]

        binary_string = shape_binary + b''.join(shape_elements_binary)
        return binary_string

    def decode_shape(self, binary_data: bytes):
        """
        Decodes the shape of a numpy array from a binary string
        :param binary_data:
        :return:
        """
        shape_length = int.from_bytes(binary_data[:4], byteorder='big')
        shape_elements = [int.from_bytes(binary_data[4 + 4 * i: 4 + 4 * (i + 1)], byteorder='big') for i in range(shape_length)]
        binary_data_without_shape = binary_data[4 + 4 * shape_length:]

        return tuple(shape_elements), binary_data_without_shape

    def encode(self, array_to_save: np.ndarray):
        if array_to_save.dtype == np.bool:
            encoded_shape = self.encode_shape(array_to_save.shape)
            bit_packed_array = np.packbits(array_to_save)
            output = io.BytesIO()
            np.save(output, bit_packed_array)

            binary_string = self.magic_numbers[1] + encoded_shape + output.getvalue()
        else:
            output = io.BytesIO()
            np.save(output, array_to_save)
            binary_string = self.magic_numbers[0] + output.getvalue()

        return binary_string

    def decode(self, value: Binary) -> np.ndarray:

        # Check which magic code matches
        if value[:4] == self.magic_numbers[0]:
            value = value[4:]
            loaded_array = np.load(io.BytesIO(value))
        elif value[:4] == self.magic_numbers[1]:
            value = value[4:]
            shape, value = self.decode_shape(value)
            loaded_packed_array = np.load(io.BytesIO(value))
            loaded_array = np.unpackbits(loaded_packed_array)
            loaded_array = loaded_array.astype(np.bool)
            loaded_array = loaded_array[:np.prod(shape)]
            loaded_array = loaded_array.reshape(shape)
        else:
            raise ValueError('The magic number does not match any of the supported formats')

        return loaded_array

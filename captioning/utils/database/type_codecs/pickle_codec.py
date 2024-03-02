import pickle
from bson.binary import Binary, USER_DEFINED_SUBTYPE
from bson.codec_options import TypeDecoder


class PickleEncoderDecoder:

    magic_numbers = (b'PICK',)

    def encode(self, value):

        binary_string = self.magic_numbers[0] + pickle.dumps(value)

        return Binary(binary_string, USER_DEFINED_SUBTYPE)
    def decode(self, value):

        # Removes the magic number
        value = value[len(self.magic_numbers[0]):]

        return pickle.loads(value)


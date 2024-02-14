from bson import Binary
from bson.binary import USER_DEFINED_SUBTYPE
from bson.codec_options import TypeDecoder


class MultiTypeCodec(TypeDecoder):  # The object is only a type decoder since encoding is managed with fallback encoding

    bson_type = Binary

    def __init__(self):
        self.magic_number_to_decoder = {}
        self.type_to_encoder = {}
        self.default_encoder = None

    def register_encoder(self, type, encoder):
        """
        Registers an encoder object for known types
        :param type:
        :param encoder:
        :return:
        """
        self.type_to_encoder[type] = encoder

    def register_default_encoder(self, encoder):
        """
        Registers an encoder object for unknown types
        :param encoder:
        :return:
        """
        self.default_encoder = encoder

    def register_decoder(self, decoder):
        """
        Registers a decoder object
        :param decoder:
        :return:
        """
        for magic_number in decoder.magic_numbers:
            self.magic_number_to_decoder[magic_number] = decoder

    def __call__(self, value) -> Binary:
        value_type = type(value)
        if value_type in self.type_to_encoder:
            encoder = self.type_to_encoder[value_type]
        else:
            encoder = self.default_encoder

        encoded_value = Binary(encoder.encode(value), USER_DEFINED_SUBTYPE)
        return encoded_value

    def transform_bson(self, value):
        if value.subtype != USER_DEFINED_SUBTYPE:
            # The value is a plain binary value
            return value

        magic_number_length = len(next(iter(self.magic_number_to_decoder)))
        magic_number = value[:magic_number_length]

        if magic_number in self.magic_number_to_decoder:
            decoder = self.magic_number_to_decoder[magic_number]
            return decoder.decode(value)
        else:
            raise ValueError("Unknown magic number: {}".format(magic_number))


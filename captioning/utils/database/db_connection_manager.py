import urllib
from typing import Dict

import numpy as np
import pymongo
from bson.codec_options import CodecOptions, TypeRegistry
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.read_concern import ReadConcern

from utils.database.type_codecs.multitype_codec import MultiTypeCodec
from utils.database.type_codecs.numpy_codec import NumpyEncoderDecoder
from utils.database.type_codecs.pickle_codec import PickleEncoderDecoder


class DBConnectionManager:
    """
    Helper class for managing the database connection
    """
    def __init__(self, connection_configuration: Dict):
        """
        Initializes the connection manager with the parameters used to instantiate connections
        :param connection_configuration:
        """
        self.host = connection_configuration["host"]
        self.port = None
        if "port" in connection_configuration:
            self.port = connection_configuration["port"]
        self.username = None
        if "username" in connection_configuration:
            self.username = connection_configuration["username"]
        self.password = None
        if "password_file" in connection_configuration:
            self.password = open(connection_configuration["password_file"], "r").read().strip()
        if "password" in connection_configuration:
            self.password = connection_configuration["password"]
        self.options = {}
        if "options" in connection_configuration:
            self.options = connection_configuration["options"]

        self.database_name = connection_configuration["database_name"]

    def build_connection_string(self) -> str:
        connection_string = "mongodb://"
        if self.username is not None:
            connection_string += urllib.parse.quote_plus(self.username)
            if self.password is not None:
                connection_string += ":" + urllib.parse.quote_plus(self.password)
            connection_string += "@"

        if isinstance(self.host, list):
            hosts_count = len(self.host)
            for i in range(hosts_count):
                connection_string += self.host[i]
                if self.port is not None:
                    connection_string += ":" + str(self.port[i])
                if i < hosts_count - 1:
                    connection_string += ","
        else:
            connection_string += self.host
            if self.port is not None:
                connection_string += ":" + str(self.port)

        # Adds the options if they are present
        if len(self.options) > 0:
            connection_string += "/?"
            for key, value in self.options.items():
                connection_string += key + "=" + value + "&"
            connection_string = connection_string[:-1]

        return connection_string

    def get_new_connection(self) -> MongoClient:
        """
        Creates a new connection to the database
        :return:
        """
        return MongoClient(self.build_connection_string())

    def get_database_name(self) -> str:
        """
        Gets the name for the default database
        :return:
        """
        return self.database_name

    def get_database(self, connection: MongoClient) -> Database:
        """
        Gets the default database
        :param connection:
        :return:
        """
        return connection[self.get_database_name()]

    def get_collection(self, connection: MongoClient, name: str) -> Collection:
        """
        Gets a collection with automatic binary data encoding and decoding applied
        :param name:
        :return:
        """
        # Creates the codec for our custom datatypes
        multitype_codec = MultiTypeCodec()
        numpy_codec = NumpyEncoderDecoder()
        pickle_codec = PickleEncoderDecoder()
        multitype_codec.register_encoder(np.ndarray, numpy_codec)
        multitype_codec.register_decoder(numpy_codec)
        multitype_codec.register_default_encoder(pickle_codec)
        multitype_codec.register_decoder(pickle_codec)

        codec_options = CodecOptions(type_registry=TypeRegistry([multitype_codec], fallback_encoder=multitype_codec))
        return self.get_database(connection).get_collection(name, codec_options=codec_options)

from __future__ import annotations

from typing import Dict, Generator, List

from bson import ObjectId
from pymongo import MongoClient

from utils.database.db_connection_manager import DBConnectionManager
from utils.database.db_objects.db_object import DBObject



class WebdatasetShardDBObject(DBObject):
    """
    CRUD Class for managing Webdataset Shard objects in the DB
    """
    # Name of the collection where to store the objects
    collection_name = "webdatasetShards"

    def __init__(self, _id: ObjectId, filename: str, dataset: str, contents: List[Dict]=None):
        """
        :param _id: id of the object
        :param filename: name of the shard file
        :param dataset: name of the shard dataset of which the video is part
        :param contents optional list of dictionaries listing the content of the shard associated to each file. Each dict has the following keys
            - type: image/video
            - filename: name of the file contained
            - dataset_name: name of the dataset
            - annotation_extensions: list of the webdataset annotation extensions associated to the file eg. [vidinfo.json, fine_text.json]
        """
        super().__init__(_id)

        self.filename = filename
        self.dataset = dataset
        self.contents = contents

    def __repr__(self):
        return f"WebdatasetShardDBObject(_id={self._id}, filename={self.filename}, dataset={self.dataset}, contents={self.contents})"

    @classmethod
    def find_base_metadata_by_id(cls, _id: ObjectId, connection: MongoClient, connection_manager: DBConnectionManager) -> WebdatasetShardDBObject:
        """
        Gets the base information corresponding to the given id
        :param _id: id of the shard for which to retrieve the metadata
        :param connection: the connection to the database
        :param connection_manager: the connection manager
        :return:
        """
        collection = connection_manager.get_collection(connection, cls.collection_name)

        db_representation = collection.find_one({'_id': _id}, {"contents": 0})  # Excludes the big contents structure
        db_object = cls.from_db_representation(db_representation)

        return db_object

    @classmethod
    def find_base_metadata_by_ids(cls, ids: List[ObjectId], connection: MongoClient, connection_manager: DBConnectionManager) -> Generator(WebdatasetShardDBObject):
        """
        Gets the base information corresponding to the given ids
        :param ids: ids of the shards for which to retrieve the metadata
        :param connection: the connection to the database
        :param connection_manager: the connection manager
        :return:
        """
        collection = connection_manager.get_collection(connection, cls.collection_name)

        db_representations = collection.find({'_id': {'$in': ids}}, {"contents": 0})
        db_objects = (cls.from_db_representation(db_representation) for db_representation in db_representations)

        return db_objects

    @classmethod
    def find_all_shard_ids(cls, dataset: str, connection: MongoClient, connection_manager: DBConnectionManager) -> Generator[ObjectId, None, None]:
        """
        Gets gets all shard ids
        :param connection: the connection to the database
        :param connection_manager: the connection manager
        :return:
        """
        collection = connection_manager.get_collection(connection, cls.collection_name)

        query = {}
        if dataset is not None:
            query["dataset"] = dataset

        object_ids = collection.find(query, {"_id": 1})
        object_ids = (obj_id["_id"] for obj_id in object_ids)

        return object_ids

    @classmethod
    def find_all_shard_filenames(cls, dataset: str, connection: MongoClient, connection_manager: DBConnectionManager) -> Generator[str, None, None]:
        """
        Gets gets all shard filenames
        :param connection: the connection to the database
        :param connection_manager: the connection manager
        :return:
        """
        collection = connection_manager.get_collection(connection, cls.collection_name)

        query = {}
        if dataset is not None:
            query["dataset"] = dataset

        object_filenames = collection.find(query, {"_id": 1, "filename": 1})
        object_filenames = (object_filename["filename"] for object_filename in object_filenames)

        return object_filenames



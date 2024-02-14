from __future__ import annotations

from typing import Dict, Generator

from bson import ObjectId
from pymongo import MongoClient

from utils.database.db_connection_manager import DBConnectionManager
from utils.database.db_objects.db_object import DBObject


class VideoDBObject(DBObject):
    """
    CRUD Class for managing Video objects in the DB
    """
    # Name of the collection where to store the objects
    collection_name = "videos"

    def __init__(self, _id: ObjectId, filename: str, dataset: str, frames_count: int, framerate: float, annotation_metadata: Dict, annotation: Dict):
        """
        :param _id: id of the object
        :param filename: name of the file
        :param dataset: name of the dataset of which the video is part
        :param frames_count: number of frames in the video
        :param framerate: framerate of the video
        :param annotation_metadata: dictionary mapping each annotation data type to the corresponding metadata.
        :param annotation: dictionary mapping each annotation data type to the corresponding video-level annotation
        """
        super().__init__(_id)

        self.filename = filename
        self.dataset = dataset
        self.frames_count = frames_count
        self.framerate = framerate
        self.annotation_metadata = annotation_metadata
        self.annotation = annotation

    def __repr__(self):
        return f"VideoDBObject(_id={self._id}, filename={self.filename}, dataset={self.dataset}, frames_count={self.frames_count}, framerate={self.framerate}, annotation_metadata={self.annotation_metadata}, annotation={self.annotation})"

    @classmethod
    def find_base_metadata_by_id(cls, _id: ObjectId, connection: MongoClient, connection_manager: DBConnectionManager) -> Dict:
        """
        Gets the filename corresponding to the given id
        :param _id: id of the video for which to retrieve the metadata
        :param connection: the connection to the database
        :param connection_manager: the connection manager
        :return:
        """
        collection = connection_manager.get_collection(connection, cls.collection_name)

        db_representation = collection.find_one({'_id': _id}, {"annotation_metadata": 0, "annotation": 0})  # Excludes the big annotation metadata structure

        return db_representation

    @classmethod
    def find_all_video_ids(cls, connection: MongoClient, connection_manager: DBConnectionManager) -> Generator[ObjectId, None, None]:
        """
        Gets the filename corresponding to the given id
        :param connection: the connection to the database
        :param connection_manager: the connection manager
        :return:
        """
        collection = connection_manager.get_collection(connection, cls.collection_name)

        object_ids = collection.find({}, {"_id": 1})
        object_ids = (obj_id["_id"] for obj_id in object_ids)

        return object_ids

    @classmethod
    def find_by_filename_and_dataset(cls, filename: str, dataset: str, connection: MongoClient, connection_manager: DBConnectionManager) -> VideoDBObject:
        """
        Gets the video corresponding to the given filename and dataset
        :param filename: name of the file
        :param dataset: name of the dataset
        :param connection: the connection to the database
        :param connection_manager: the connection manager
        :return:
        """
        collection = connection_manager.get_collection(connection, cls.collection_name)

        db_representation = collection.find_one({'filename': filename, 'dataset': dataset})
        db_object = VideoDBObject.from_db_representation(db_representation)

        return db_object



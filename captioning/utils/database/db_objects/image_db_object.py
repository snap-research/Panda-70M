from __future__ import annotations

from typing import Dict

from bson import ObjectId
from pymongo import MongoClient

from utils.database.db_connection_manager import DBConnectionManager
from utils.database.db_objects.db_object import DBObject


class ImageDBObject(DBObject):
    """
    CRUD Class for managing Image objects in the DB
    """
    # Name of the collection where to store the objects
    collection_name = "images"

    def __init__(self, _id: ObjectId, filename: str, dataset: str, annotation: Dict):
        """
        :param _id: id of the object
        :param filename: name of the file
        :param dataset: name of the dataset of which the image is part
        :param annotation: annotation of the frame expressed as a dictionary
                           the dictionary has an entry for each annotation type that is present in the current frame
        """
        super().__init__(_id)

        self.filename = filename
        self.dataset = dataset
        self.annotation = annotation

    def __repr__(self):
        return f"ImageDBObject(_id={self._id}, filename={self.filename}, dataset={self.dataset}, annotation={self.annotation})"

    @classmethod
    def find_by_filename_and_dataset(cls, filename: str, dataset: str, connection: MongoClient, connection_manager: DBConnectionManager) -> ImageDBObject:
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
        db_object = ImageDBObject.from_db_representation(db_representation)

        return db_object

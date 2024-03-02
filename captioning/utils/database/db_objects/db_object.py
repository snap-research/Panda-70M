from __future__ import annotations

import fractions
from typing import List, Generator

import pymongo
from bson import ObjectId
from pymongo import MongoClient
from pymongo.results import DeleteResult, InsertManyResult, InsertOneResult, UpdateResult

from utils.database.db_connection_manager import DBConnectionManager


class DBObject:

    def __init__(self, _id: ObjectId):
        if _id is None:
            _id = ObjectId()
        self._id = _id

    def delete(self, connection: MongoClient, connection_manager: DBConnectionManager) -> DeleteResult:
        """
        Deletes the object from the database
        :param connection: the connection to the database
        :param connection_manager: the connection manager
        :return:
        """
        collection = connection_manager.get_collection(connection, self.collection_name)

        return collection.delete_one({'_id': self._id})

    def update(self, connection: MongoClient, connection_manager: DBConnectionManager) -> UpdateResult:
        """
        Updates the object in the database
        :param connection: the connection to the database
        :param connection_manager: the connection manager
        :return:
        """
        collection = connection_manager.get_collection(connection, self.collection_name)

        db_representation = self.to_db_representation()
        return collection.update_one({'_id': self._id}, {"$set": db_representation})

    def create(self, connection: MongoClient, connection_manager: DBConnectionManager) -> InsertOneResult:
        """
        Inserts the object in the database
        :param connection: the connection to the database
        :param connection_manager: the connection manager
        :return:
        """
        collection = connection_manager.get_collection(connection, self.collection_name)
        db_representation = self.to_db_representation()
        return collection.insert_one(db_representation)

    def to_db_representation(self) -> dict:
        """
        Returns the representation of the object that can be stored in the database
        :return:
        """
        representation = {}
        for key, value in self.__dict__.items():
            if key not in ["collection_name"]:
                representation[key] = value
        return representation

    @classmethod
    def from_db_representation(cls, representation: dict):
        """
        Creates an object from the representation stored in the database
        :param representation: the representation of the object in the database
        :return:
        """
        if representation is None:
            return None

        return cls(**representation)

    @classmethod
    def find_by_id(cls, _id: ObjectId, connection: MongoClient, connection_manager: DBConnectionManager) -> DBObject:
        """
        Finds an object by its id
        :param _id: the id of the object
        :param connection: the connection to the database
        :param connection_manager: the connection manager
        :return: the object associated to the id or None if it does not exist
        """
        collection = connection_manager.get_collection(connection, cls.collection_name)

        db_representation = collection.find_one({'_id': _id})
        if db_representation is None:
            return None

        db_object = cls.from_db_representation(db_representation)
        return db_object

    @classmethod
    def find_all(cls, connection: MongoClient, connection_manager: DBConnectionManager) -> Generator[DBObject]:
        """
        Finds all objects
                :param connection: the connection to the database
        :param connection_manager: the connection manager
        :return: all the objects as a generator
        """
        collection = connection_manager.get_collection(connection, cls.collection_name)

        db_representations = collection.find()
        db_objects = (cls.from_db_representation(db_representation) for db_representation in db_representations)

        return db_objects

    @classmethod
    def find_by_ids(cls, ids: List[ObjectId], connection: MongoClient, connection_manager: DBConnectionManager) -> Generator[DBObject]:
        """
        Finds objects by their ids
        :param ids: the ids of the objects
        :param connection: the connection to the database
        :param connection_manager: the connection manager
        :return: the objects matching any of the ids as a generator. Objects are unordered
        """
        collection = connection_manager.get_collection(connection, cls.collection_name)

        db_representations = collection.find({'_id': {'$in': ids}})
        db_objects = (cls.from_db_representation(db_representation) for db_representation in db_representations)

        return db_objects

    @staticmethod
    def create_many(objects: List[DBObject], connection: MongoClient, connection_manager: DBConnectionManager) -> InsertManyResult:
        """
        Creates objects in batch in the database. Objects must refer all to the same collection
        :param objects: the objects to create
        :param connection: the connection to the database
        :param connection_manager: the connection manager
        :return:
        """
        first_collection_name = objects[0].collection_name
        for obj in objects:
            if obj.collection_name != first_collection_name:
                raise ValueError("Cannot create objects in batch if they refer to different collections: {} and {}".format(first_collection_name, obj.collection_name))

        collection = connection_manager.get_collection(connection, objects[0].collection_name)

        db_representations = [obj.to_db_representation() for obj in objects]
        return collection.insert_many(db_representations)

    @staticmethod
    def delete_many(objects: List[DBObject], connection: MongoClient, connection_manager: DBConnectionManager) -> DeleteResult:
        """
        Deletes objects in batch from the database. Objects must refer all to the same collection
        :param objects: the objects to delete
        :param connection: the connection to the database
        :param connection_manager: the connection manager
        :return:
        """
        first_collection_name = objects[0].collection_name
        for obj in objects:
            if obj.collection_name != first_collection_name:
                raise ValueError("Cannot delete objects in batch if they refer to different collections: {} and {}".format(first_collection_name, obj.collection_name))

        collection = connection_manager.get_collection(connection, objects[0].collection_name)

        return collection.delete_many({'_id': {'$in': [obj._id for obj in objects]}})

    @classmethod
    def delete_by_id(cls, _id: ObjectId, connection: MongoClient, connection_manager: DBConnectionManager) -> DeleteResult:
        """
        Deletes the object with the given id
        :param _id:
        :param connection:
        :param connection_manager:
        :return:
        """
        collection = connection_manager.get_collection(connection, cls.collection_name)
        return collection.delete_one({'_id': _id})

    @classmethod
    def delete_by_ids(cls, ids: List[ObjectId], connection: MongoClient, connection_manager: DBConnectionManager) -> DeleteResult:
        """
        Deletes the objects with the given ids
        :param ids:
        :param connection:
        :param connection_manager:
        :return:
        """
        collection = connection_manager.get_collection(connection, cls.collection_name)
        return collection.delete_many({'_id': {'$in': ids}})


def main():
    import numpy as np
    from utils.database.db_objects.video_db_object import VideoDBObject

    connection_configuration = {
        "host": "127.0.0.1",
        "database_name": "db_objects_test"
    }

    connection_manager = DBConnectionManager(connection_configuration)
    connection = connection_manager.get_new_connection()

    annotation_metadata = {
        "bounding_boxes": np.array([[0, 0, 100, 100], [100, 100, 200, 200]]),
        "binary_bounding_boxes": np.array([[False, False, True, True], [True, False, False, True]], dtype=np.bool),
        "custom_type": fractions.Fraction(1, 20),
    }

    annotation = {
        "text": "This is a text description of the video"
    }

    video_db_object = VideoDBObject(None, "filename.mp4", "mydataset", 125, 29.97, annotation_metadata, annotation)
    video_db_object.create(connection, connection_manager)
    retrieved_video_db_object = VideoDBObject.find_by_id(video_db_object._id, connection, connection_manager)

    video_db_object.delete(connection, connection_manager)

    video_db_object.create(connection, connection_manager)
    video_db_object.framerate = 200.0
    video_db_object.update(connection, connection_manager)
    video_db_object.delete(connection, connection_manager)

    video_db_object.create(connection, connection_manager)
    VideoDBObject.delete_by_id(video_db_object._id, connection, connection_manager)

    all_video_db_objects = []
    for i in range(10):
        all_video_db_objects.append(VideoDBObject(None, "filename{}.mp4".format(i), "mydataset", 125, 29.97, annotation_metadata, annotation))
    VideoDBObject.create_many(all_video_db_objects, connection, connection_manager)
    VideoDBObject.delete_many(all_video_db_objects, connection, connection_manager)

    VideoDBObject.create_many(all_video_db_objects, connection, connection_manager)
    VideoDBObject.delete_by_ids([obj._id for obj in all_video_db_objects], connection, connection_manager)

    print("All done")

if __name__ == "__main__":

    main()



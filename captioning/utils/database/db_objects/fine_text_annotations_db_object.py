from __future__ import annotations

from typing import Dict, List

from bson import ObjectId
from pymongo import MongoClient

from utils.database.db_connection_manager import DBConnectionManager
from utils.database.db_objects.db_object import DBObject


class FineTextAnnotationsDBObject(DBObject):
    """
    CRUD Class for managing fine text annotation objects in the DB
    """
    # Name of the collection where to store the objects
    collection_name = "fine_text_annotations"

    def __init__(self, _id: ObjectId, video_id: ObjectId, annotation: List[Dict]):
        """
        :param _id: id of the object
        :param video_id: id of the video to which the annotation belongs
        :param annotation: list of annotations specifying start timestamp, end timestamp, and text for each action in the video
        """
        super().__init__(_id)
        self.video_id = video_id
        self.annotation = annotation

    def __repr__(self):
        return f"FineTextAnnotationsDBObject(_id={self._id}, video_id={self.video_id}, annotation={self.annotation})"

    @classmethod
    def find_by_video_id(cls, video_id: ObjectId, connection: MongoClient, connection_manager: DBConnectionManager) -> FineTextAnnotationsDBObject:
        """
        Finds an object by the corresponding video id
        :param video_id: the id of the corrsponding video object
        :return: the object associated to the id or None if it does not exist
        """
        collection = connection_manager.get_collection(connection, cls.collection_name)

        db_representation = collection.find_one({'video_id': video_id})
        if db_representation is None:
            return None

        db_object = cls.from_db_representation(db_representation)
        return db_object


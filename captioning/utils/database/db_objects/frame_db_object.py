from __future__ import annotations
from typing import Dict, List

from bson import ObjectId
from pymongo import MongoClient

from utils.database.db_connection_manager import DBConnectionManager
from utils.database.db_objects.db_object import DBObject


class FrameDBObject(DBObject):
    """
    CRUD Class for managing Frame objects in the DB
    """
    # Name of the collection where to store the objects
    collection_name = "frames"

    def __init__(self, _id: ObjectId, video_id: ObjectId, frame_number: int, annotation: Dict):
        """
        :param _id: id of the object
        :param video_id: id of the video
        :param frame_number: number of the frame in the video starting from 0
        :param annotation: annotation of the frame expressed as a dictionary
                           the dictionary has an entry for each annotation type that is present in the current frame
        """
        super().__init__(_id)

        self.video_id = video_id
        self.frame_number = frame_number
        self.annotation = annotation

    def __repr__(self):
        return f"FrameDBObject(_id={self._id}, video_id={self.video_id}, frame_number={self.frame_number}, annotation={self.annotation})"

    @classmethod
    def find_by_video_id_and_frame_numbers(cls, video_id: ObjectId, frame_numbers: List[int], connection: MongoClient, connection_manager: DBConnectionManager) -> Generator[FrameDBObject, None, None]:
        """
        Gets the frames corresponding to the given video and frame numbers
        :param video_id: id of the video
        :param frame_numbers: numbers of the frames in the video to retrieve
        :param connection: the connection to the database
        :param connection_manager: the connection manager
        :return:
        """
        # TODO create an index on video_id,frame_number

        collection = connection_manager.get_collection(connection, cls.collection_name)
        db_representations = collection.find({'video_id': video_id, 'frame_number': {'$in': frame_numbers}})

        db_objects = (cls.from_db_representation(db_representation) for db_representation in db_representations)
        return db_objects




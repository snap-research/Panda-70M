import numpy as np
import json
import glob
import os
import hashlib
from pathlib import Path
import gzip
import pickle as pkl

from typing import Dict, List, Tuple, Set
from utils.text.text_embedding_storages.text_embedding_storage import TextEmbeddingStorage

class DirectoryTextEmbeddingStorage(TextEmbeddingStorage):

    def __init__(self, storage_config: Dict):
        super().__init__(storage_config)

        self.root_directory = storage_config["root_directory"]
        self.embeddings_transform = storage_config["embeddings_transform"]

        # Creates the root directory if it does not exist
        Path(self.root_directory).mkdir(parents=True, exist_ok=True)

        # Initializes the storage from the disk
        self.text_metadata = {}
        self.load_text_metadata()

    def add_file_to_text_metadata(self, json_file: str):
        """
        Adds the given json file to the text metadata
        """

        with open(json_file, "r") as input_file:
            current_json_data = json.load(input_file)

        current_text = current_json_data["text"]
        self.text_metadata[current_text] = current_json_data

    def load_text_metadata(self):
        """
        Gets metadata on all text from the storage
        """

        all_json_files = list(sorted(glob.glob(os.path.join(self.root_directory, "*.json"))))
        for current_json_file in all_json_files:
            self.add_file_to_text_metadata(current_json_file)

    def add(self, text: str, embeddings: np.ndarray, eot_location: int, token_ids: List[int], language_model_name: str):
        """
        Adds the given embeddings to the storage
        :param text: The text associated to the embeddings
        :param embeddings: The embeddings to store
        :param eot_location: The location of the end of text token
        :param token_ids: The token ids associated to the embeddings
        :param language_model_name: The name of the language model used to compute the embeddings
        """
        
        if text in self.text_metadata:
            raise ValueError("Text {} already in the storage".format(text))

        cut_embeddings = embeddings
        embeddings_length = embeddings.shape[0]
        if eot_location < embeddings_length - 1:
            cut_embeddings = cut_embeddings[:eot_location+1]

        current_results = {
            "embeddings": cut_embeddings,
            "eot_location": eot_location,
            "token_ids": token_ids,
            "model_name": language_model_name
        }

        # Writes the text
        text_filename = self.get_text_filename_by_text(text)
        text_data = {
            "text": text,
        }
        with open(text_filename, "w") as output_file:
            json.dump(text_data, output_file)

        # Writes the embeddings
        embeddings_filename = self.get_embeddings_filename_by_text(text)
        with gzip.open(embeddings_filename, "wb") as output_file:
            pkl.dump(current_results, output_file)

        self.add_file_to_text_metadata(text_filename)

    def __getitem__(self, text: str) -> Dict:
        """
        Retrieves the embeddings associated to the given text
        """

        if text not in self.text_metadata:
            raise ValueError("Text {} not in the storage".format(text))

        text_embeddings_filename = self.get_embeddings_filename_by_text(text)
        with gzip.open(text_embeddings_filename, "rb") as input_file:
            text_embeddings_data = pkl.load(input_file)

        embedding = text_embeddings_data["embeddings"]
        eot_location = text_embeddings_data["eot_location"]
        token_ids = text_embeddings_data["token_ids"]
        language_model_name = text_embeddings_data["model_name"]

        transformed_embedding, transformed_eot_location = self.embeddings_transform(embedding, eot_location)
        return transformed_embedding

    def get_all_texts(self) -> Set[str]:
        """
        Retrieves all the captions stored in the storage
        """

        return set(self.text_metadata.keys())

    def get_hash_by_text(self, text: str) -> str:
        """
        Retrieves the hash associated to the given text
        """

        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def get_embeddings_filename_by_text(self, text: str) -> str:
        """
        Retrieves the embeddings filename associated to the given text
        """

        current_hash = self.get_hash_by_text(text)
        embedding_output_filename = os.path.join(self.root_directory, current_hash + ".pkl.gz")
        return embedding_output_filename

    def get_text_filename_by_text(self, text: str) -> str:
        """
        Retrieves the text filename associated to the given text
        """

        current_hash = self.get_hash_by_text(text)
        text_output_filename = os.path.join(self.root_directory, current_hash + ".json")
        return text_output_filename




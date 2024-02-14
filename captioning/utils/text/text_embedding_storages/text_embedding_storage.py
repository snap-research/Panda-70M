import numpy as np

from typing import Dict, List, Tuple

class TextEmbeddingStorage:

    def __init__(self, storage_config: Dict):
        self.storage_config = storage_config

    def add(self, text: str, embeddings: np.ndarray, eot_location: int, token_ids: List[int], language_model_name: str):
        """
        Adds the given embeddings to the storage
        :param text: The text associated to the embeddings
        :param embeddings: The embeddings to store
        :param eot_location: The location of the end of text token
        :param token_ids: The token ids associated to the embeddings
        :param language_model_name: The name of the language model used to compute the embeddings
        """
        
        raise NotImplementedError()

    def __getitem__(self, text: str) -> Dict:
        """
        Retrieves the embeddings associated to the given text
        """

        raise NotImplementedError()

    def get_all_texts(self) -> List[str]:
        """
        Retrieves all the captions stored in the storage
        """

        raise NotImplementedError()

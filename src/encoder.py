import numpy as np
from tensorflow.keras import models

from src.config import Configs


class _ModelLoader:
    """ For transporting a neural network to a working object of Encoder class"""

    def __init__(self):
        self.__model_name = Configs.get('models.encoder.name')
        self.__model_path = f'{Configs.get("directories.models")}/{self.__model_name}/encoder'
        self.__model = None

    def get(self):
        """ Get encoder from RAM or local file storage """
        if self.__model is None:
            self.__model = self.__load()
        return self.__model

    def __load(self):
        """  Load model from local file storage """
        return models.load_model(self.__model_path)


class ModelEncoder:
    """ For data compression from (n, 768) shape vectors to (n, 256) shape """

    def __init__(self):
        self.__model_loader = _ModelLoader()

    def predict(self, vectors: np.ndarray, *args, **kwargs) -> np.ndarray:
        """ Compress vectors of (n, 768) shape to (n, 256) shape """
        return self.__model_loader.get().predict(vectors, *args, **kwargs).reshape(1, -1)


""" TODO: we have encoder loader and we need saver now """

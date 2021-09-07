import torch
import numpy as np

from config import Configs


class _ModelLoader:
    """ For transporting a neural network to a working object of Encoder class"""

    def __init__(self):
        self.__model_name = Configs.get('models.encoder.name')
        self.__model_path = f'{Configs.get("directories.models")}/{self.__model_name}.pickle'
        self.__model = None

    def get(self):
        """ Get encoder from RAM or local file storage """

        if self.__model is None:
            self.__model = self.__load()

        # TODO:
        # if torch.cuda.is_available():
        #     self.__model.cuda()

        self.__model.eval()

        return self.__model

    def __load(self):
        """  Load model from local file storage """

        self.__model = torch.load(self.__model_path)
        return self.__model


class ModelVitEncoder:
    """ For data compression from (n, 768) shape vectors to (n, 256) shape """

    def __init__(self):
        self.__model_loader = _ModelLoader()

    def predict(self, vectors: np.ndarray) -> np.ndarray:
        """ Compress vectors of (n, 768) shape to (n, 256) shape """

        # TODO: check correctness
        # TODO: add GPU usage

        vectors = torch.Tensor(vectors)

        answer = self.__model_loader.get()(vectors)
        answer = answer.detach().numpy()

        return answer


""" TODO: we have encoder loader and we need saver now """

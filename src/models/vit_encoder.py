from src.exceptions import NeuralNetworkModelNotFoundError
from src.models.abstract import ModelLoader
from config import (
    MODEL_VIT_ENCODER_PATH,
)


class ModelLoaderViTEncoder(ModelLoader):
    __path = MODEL_VIT_ENCODER_PATH
    __model = None

    @classmethod
    def __make(cls):
        raise NeuralNetworkModelNotFoundError

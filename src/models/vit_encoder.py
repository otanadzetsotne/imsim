from src.exceptions import NeuralNetworkModelNotFoundError
from src.models.abstract import ModelLoader
from config import (
    MODEL_VIT_ENCODER_PATH,
)


class ModelLoaderViTEncoder(ModelLoader):
    _path = MODEL_VIT_ENCODER_PATH
    _model = None

    @classmethod
    def _make(cls):
        raise NeuralNetworkModelNotFoundError

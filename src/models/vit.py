from pytorch_pretrained_vit import ViT
from pytorch_pretrained_vit.model import ViT as modelViT

from src.models.abstract import ModelLoader
from config import (
    MODEL_INPUT,
    MODEL_VIT_NAME,
    MODEL_VIT_PATH,
)


class ModelLoaderViT(ModelLoader):
    """
    For transporting a neural network to a working object of Model class
    """

    _path = MODEL_VIT_PATH
    _model = None

    @classmethod
    def _make(cls) -> modelViT:
        """
        Download model from ViT library to local file storage
        :return: ViT model
        """

        return ViT(
            name=MODEL_VIT_NAME,
            image_size=MODEL_INPUT,
            pretrained=True,
        )

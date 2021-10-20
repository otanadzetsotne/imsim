from pytorch_pretrained_vit import ViT
from pytorch_pretrained_vit.model import ViT as modelViT

from src.models.abstract import ModelLoader
from config import (
    MODEL_INPUT,
    MODEL_INPUT_TINY,
    MODEL_VIT_TYPE,
    MODEL_VIT_PATH,
    MODEL_VIT_TINY_PATH,
)


class _ModelLoaderViTBase(ModelLoader):
    """
    For transporting a neural network to a working object of Model class
    """

    _type = MODEL_VIT_TYPE
    _path = None
    _model = None
    _image_size = None

    @classmethod
    def _make(cls) -> modelViT:
        """
        Download model from ViT library to local file storage
        :return: ViT model
        """

        return ViT(
            name=cls._type,
            image_size=cls._image_size,
            pretrained=True,
        )


class ModelLoaderViT(_ModelLoaderViTBase):
    _path = MODEL_VIT_PATH
    _image_size = MODEL_INPUT


class ModelLoaderViTTiny(_ModelLoaderViTBase):
    _path = MODEL_VIT_TINY_PATH
    _image_size = MODEL_INPUT_TINY

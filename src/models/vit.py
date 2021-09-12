import os

import torch
from pytorch_pretrained_vit import ViT
from pytorch_pretrained_vit.model import ViT as modelViT

from config import (
    MODEL_INPUT,
    MODEL_VIT_NAME,
    MODEL_VIT_PATH,
)


class ModelLoaderViT:
    """
    For transporting a neural network to a working object of Model class
    """

    __path = MODEL_VIT_PATH
    __model = None

    @classmethod
    def get(cls) -> modelViT:
        """
        Get model from RAM, file storage or download from library
        :return: ViT model
        """

        if not cls.__exists():
            cls.__model = cls.__make()

        if cls.__model is None:
            cls.__model = cls.__load()

        # if torch.cuda.is_available():
        #     cls.__model.cuda()

        cls.__model.eval()

        return cls.__model

    @classmethod
    def __load(cls) -> modelViT:
        """
        Load model from local file storage
        :return: ViT model
        """

        model = torch.load(cls.__path)

        return model

    @classmethod
    def __make(cls) -> modelViT:
        """
        Download model from ViT library to local file storage
        :return: ViT model
        """

        model = ViT(
            name=MODEL_VIT_NAME,
            image_size=MODEL_INPUT,
            pretrained=True,
        )

        torch.save(model, cls.__path)

        return model

    @classmethod
    def __exists(cls) -> bool:
        """
        Check if model is already downloaded to local file storage
        :return: bool
        """

        return os.path.exists(cls.__path) and os.path.isfile(cls.__path)

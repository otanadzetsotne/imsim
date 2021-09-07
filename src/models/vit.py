import os

import torch
from pytorch_pretrained_vit import ViT
from pytorch_pretrained_vit.model import ViT as modelViT

from config import (
    MODEL_DIR,
    MODEL_INPUT,
    MODEL_VIT_NAME,
)


class ModelLoaderViT:
    """
    For transporting a neural network to a working object of Model class
    """

    __path = f'{MODEL_DIR}/{MODEL_VIT_NAME}_{MODEL_INPUT}.pickle'
    __model = None

    @classmethod
    def get(cls) -> modelViT:
        """
        Get model from RAM, file storage or download from library
        :return: ViT model
        """

        if not cls.__downloaded():
            cls.__model = cls.__download()

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
    def __download(cls) -> modelViT:
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
    def __downloaded(cls) -> bool:
        """
        Check if model is already downloaded to local file storage
        :return: bool
        """

        return os.path.exists(cls.__path) and os.path.isfile(cls.__path)

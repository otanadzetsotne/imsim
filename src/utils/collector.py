import torch

from src.models.vit import ModelLoaderViT
from src.models.vit_encoder import ModelLoaderViTEncoder
from src.dtypes import Model
from config import (
    MODEL_VIT_NAME,
    MODEL_VIT_ENCODER_NAME,
)


class _Identity(torch.nn.Module):
    """
    Linear layer for torch model
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        return x


class Collector:
    __vit = None

    @classmethod
    def collect(
            cls,
            model_type: Model,
    ) -> torch.nn.Module:
        """
        Get model
        :param model_type: Model
        :return: torch.nn.Module
        """

        if model_type == Model.vit:
            if cls.__vit is None:
                cls.__vit = cls.__vit_collect()

            return cls.__vit

    @classmethod
    def __vit_collect(
            cls,
    ) -> torch.nn.Module:
        """
        Get ViT model
        :return: torch.nn.Module
        """

        model = torch.nn.Sequential()
        model.add_module(MODEL_VIT_NAME, cls.__get_vit())
        model.add_module(MODEL_VIT_ENCODER_NAME, cls.__get_vit_encoder())

        return model

    @classmethod
    def __get_vit(
            cls,
    ) -> torch.nn.Module:
        """
        Get base ViT model
        :return: model
        """

        # Get ViT model
        model_vit = ModelLoaderViT.get()

        # Drop classification layer
        model_vit.fc = _Identity()

        return model_vit

    @classmethod
    def __get_vit_encoder(
            cls,
    ) -> torch.nn.Module:
        """
        Get ViT model encoder
        :return: encoder model
        """

        return ModelLoaderViTEncoder.get()

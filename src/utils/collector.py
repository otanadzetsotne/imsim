import torch

from src.facades.models import Models
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
            return cls.__vit_collect()

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
        # model.add_module(MODEL_VIT_ENCODER_NAME, cls.__get_vit_encoder())

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
        model_vit = Models.vit.get()
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

        return Models.vit_encoder.get()

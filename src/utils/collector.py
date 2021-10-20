import torch

from src.facades.models import Models
from src.dtypes import Model
from config import (
    MODEL_NAME_VIT,
)


# TODO: move that to separate model
class Identity(torch.nn.Module):
    """
    Linear layer for torch model
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        return x


class CollectorDecorators:
    @staticmethod
    def classification_to_identity(func):
        def inner(*args, **kwargs):
            model = func(*args, **kwargs)
            model.fc = Identity()

            return model

        return inner


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
            return cls.collect_vit()
        if model_type == Model.vit_tiny:
            return cls.collect_vit_tiny()

    @classmethod
    def collect_vit(
            cls,
    ) -> torch.nn.Module:
        """
        Get ViT model
        :return: torch.nn.Module
        """

        model = torch.nn.Sequential()
        model.add_module(MODEL_NAME_VIT, cls.get_vit())
        # model.add_module(MODEL_VIT_ENCODER_NAME, cls.get_vit_encoder())

        return model

    @classmethod
    def collect_vit_tiny(
            cls,
    ) -> torch.nn.Module:

        model = torch.nn.Sequential()
        model.add_module(MODEL_NAME_VIT, cls.get_vit_tiny())

        return model

    @staticmethod
    @CollectorDecorators.classification_to_identity
    def get_vit() -> torch.nn.Module:
        """
        Load base ViT model
        """

        return Models.vit.get()

    @staticmethod
    @CollectorDecorators.classification_to_identity
    def get_vit_tiny() -> torch.nn.Module:
        """
        Load base ViT model with tiny input
        """

        return Models.vit_tiny.get()

    @staticmethod
    def get_vit_encoder() -> torch.nn.Module:
        """
        Get ViT model encoder
        :return: encoder model
        """

        return Models.vit_encoder.get()

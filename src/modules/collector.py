import torch
from src.models.vit import ModelLoaderViT
from src.dtypes import Model


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
    __vit_loader = ModelLoaderViT

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
            return cls.__vit_collect()

    @classmethod
    def __vit_collect(
            cls,
    ) -> torch.nn.Module:
        """
        Get ViT model
        :return: torch.nn.Module
        """

        if cls.__vit is None:
            # Get ViT model
            model_vit = cls.__vit_loader.get()

            # Drop classification layer
            model_vit.fc = _Identity()

            # Save model
            cls.__vit = model_vit

        return cls.__vit

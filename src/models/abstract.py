import os
import torch
from torch import nn
from abc import ABCMeta, abstractmethod


class ModelLoader(meta=ABCMeta):
    __path = None
    __model = None

    @classmethod
    def get(cls) -> nn.Module:
        if not cls.__exists():
            cls.__model = cls.__make()
            cls.__save()

        if cls.__model is None:
            cls.__model = cls.__load()

        # if torch.cuda.is_available():
        #     cls.__model.cuda()

        cls.__model.eval()

        return cls.__model

    @classmethod
    def __save(cls):
        """
        Save model to local file system
        :return: None
        """

        torch.save(cls.__model, cls.__path)

    @classmethod
    def __exists(cls) -> bool:
        """
        Check if model is already downloaded to local file storage
        :return: bool
        """

        return os.path.exists(cls.__path) and os.path.isfile(cls.__path)

    @classmethod
    def __load(cls) -> nn.Module:
        """
        Load model from local file storage
        :return: nn.Module
        """

        return torch.load(cls.__path)

    @classmethod
    @abstractmethod
    def __make(cls) -> nn.Module:
        """
        Load from
        :return: nn.Module
        """

        raise NotImplementedError

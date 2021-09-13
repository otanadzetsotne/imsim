import os
import torch
from torch import nn
from abc import ABC, abstractmethod


class ModelLoader(ABC):
    _path = None
    _model = None

    @classmethod
    def get(cls) -> nn.Module:
        if not cls._exists():
            cls._model = cls._make()
            cls._save()

        if cls._model is None:
            cls._model = cls._load()

        # if torch.cuda.is_available():
        #     cls._model.cuda()

        cls._model.eval()
        cls._model.cpu()

        return cls._model

    @classmethod
    def _save(cls):
        """
        Save model to local file system
        :return: None
        """

        torch.save(cls._model, cls._path)

    @classmethod
    def _exists(cls) -> bool:
        """
        Check if model is already downloaded to local file storage
        :return: bool
        """

        return os.path.exists(cls._path) and os.path.isfile(cls._path)

    @classmethod
    def _load(cls) -> nn.Module:
        """
        Load model from local file storage
        :return: nn.Module
        """

        return torch.load(cls._path)

    @classmethod
    @abstractmethod
    def _make(cls) -> nn.Module:
        """
        Load from
        :return: nn.Module
        """

        raise NotImplementedError

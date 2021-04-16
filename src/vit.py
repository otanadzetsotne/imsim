import os
import torch
import pickle
import numpy as np
from PIL import Image
from pytorch_pretrained_vit import ViT
from pytorch_pretrained_vit.model import ViT as modelViT
from torchvision import transforms
from torchvision.transforms.transforms import Compose as transformCompose

import config as c


class _Identity(torch.nn.Module):
    """ This class replaces the layer of the neural network which is needed for classification """

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        return x


class _ModelLoader:
    """ For transporting a neural network to a working object of Model class"""

    def __init__(self, model, *args, **kwargs):
        self.__model_name = model
        self.__model_path = f'{c.path_model}/{model}.pickle'
        self.__model = None

        self.__args = args
        self.__kwargs = kwargs

    def get(self) -> modelViT:
        """ Get model from RAM, file storage or download from library """

        if not self.__downloaded():
            self.__model = self.__download()

        if self.__model is None:
            self.__model = self.__load()

        return self.__model

    def __load(self) -> modelViT:
        """  Load model from local file storage """

        with open(self.__model_path, 'rb') as f:
            model = pickle.load(f)

        return model

    def __download(self) -> modelViT:
        """ Download model from ViT library to local file storage """

        model = ViT(self.__model_name, *self.__args, **self.__kwargs)
        with open(self.__model_path, 'wb') as f:
            pickle.dump(model, f)

        return model

    def __downloaded(self) -> bool:
        """ Check if model is already downloaded to local file storage """

        return os.path.exists(self.__model_path) and os.path.isfile(self.__model_path)


class ModelViT:
    def __init__(self, model: str = 'B_16_imagenet1k', max_workers: int = 32, *args, **kwargs):
        self.__model_loader = _ModelLoader(model, *args, **kwargs)
        self.__max_workers = max_workers

    def predict(self, img: type(Image)) -> np.ndarray:
        """ Predict """
        img = self.__transform()(img).unsqueeze(0)
        with torch.no_grad():
            return np.array(self.__predictor()(img)).reshape(1, -1)

    def __classifier(self) -> modelViT:
        """ Get ViT classifier network """
        return self.__model_loader.get()

    def __predictor(self) -> modelViT:
        """ Get ViT classifier network without classifier layer """
        classifier = self.__classifier()
        classifier.fc = _Identity()
        return classifier

    @staticmethod
    def __transform() -> transformCompose:
        """ Get transform layer for neural network input """
        return transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ])

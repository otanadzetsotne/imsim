import os

import numpy as np

import torch
from torchvision import transforms
from torchvision.transforms.transforms import Compose as transformCompose

from pytorch_pretrained_vit import ViT
from pytorch_pretrained_vit.model import ViT as modelViT

from config import (
    MODEL_DIR,
    MODEL_VIT_INPUT,
    MODEL_VIT_NAME,
)
from src.datatypes import PILImage


class _Identity(torch.nn.Module):
    """ This class replaces the layer of the neural network which is needed for classification """

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        return x


class _ModelLoader:
    """ For transporting a neural network to a working object of Model class"""

    def __init__(self, name, image_size):
        self.__model_name = name
        self.__image_size = image_size

        self.__model_path = f'{MODEL_DIR}/{self.__model_name}_{self.__image_size}.pickle'
        self.__model = None

    def get(self) -> modelViT:
        """ Get model from RAM, file storage or download from library """

        if not self.__downloaded():
            self.__model = self.__download()

        if self.__model is None:
            self.__model = self.__load()

        if torch.cuda.is_available():
            self.__model.cuda()

        self.__model.eval()

        return self.__model

    def __load(self) -> modelViT:
        """  Load model from local file storage """

        # TODO: нужно проверить
        model = torch.load(self.__model_path)

        # with open(self.__model_path, 'rb') as f:
        #     model = pickle.load(f)

        return model

    def __download(self) -> modelViT:
        """ Download model from ViT library to local file storage """

        model = ViT(
            name=self.__model_name,
            image_size=self.__image_size,
            pretrained=True,
        )

        # TODO: нужно проверить
        torch.save(model, self.__model_path)

        # with open(self.__model_path, 'wb') as f:
        #     pickle.dump(model, f)

        return model

    def __downloaded(self) -> bool:
        """ Check if model is already downloaded to local file storage """

        return os.path.exists(self.__model_path) and os.path.isfile(self.__model_path)


class ModelViT:
    __model_loader = None

    def __init__(self):
        self.__model_loader = _ModelLoader(MODEL_VIT_NAME, MODEL_VIT_INPUT)

        # TODO: зачем вызывать модель заранее?
        # self.__model_loader.get()

    def predict(self, img: PILImage) -> np.ndarray:
        """ Predict """

        # Transform images to Torch tensors
        img = self.__transform()(img).unsqueeze(0)
        # Throw tensors to GPU if available
        img = img.cuda() if torch.cuda.is_available() else img

        # With disabled gradient calculation
        with torch.no_grad():
            # Make prediction
            prediction = self.__predictor()(img)
            prediction = prediction.cpu() if torch.cuda.is_available() else prediction
            prediction = np.array(prediction).reshape(1, -1)

            return prediction

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
            transforms.Resize((MODEL_VIT_INPUT, MODEL_VIT_INPUT)),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ])

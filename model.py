import os
import PIL
import torch
import pickle
import requests
from typing import Union
from pytorch_pretrained_vit import ViT
from pytorch_pretrained_vit.model import ViT as modelViT
from torchvision import transforms
from torchvision.transforms.transforms import Compose as transformCompose

import config as c


class _Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        return x


class _ModelLoader:
    def __init__(self, model, *args, **kwargs):
        self.__model_name = model
        self.__model_path = f'{c.path_model}/{model}.pickle'
        self.__model = None

        self.__args = args
        self.__kwargs = kwargs

    def get(self):
        if not self.__downloaded():
            self.__model = self.__download()

        if self.__model is None:
            self.__model = self.__load()

        return self.__model

    def __load(self):
        with open(self.__model_path, 'rb') as f:
            model = pickle.load(f)

        return model

    def __download(self):
        model = ViT(self.__model_name, *self.__args, **self.__kwargs)
        with open(self.__model_path, 'wb') as f:
            pickle.dump(model, f)

        return model

    def __downloaded(self):
        return os.path.exists(self.__model_path) and os.path.isfile(self.__model_path)


class Model:
    def __init__(self, model: str = 'B_16_imagenet1k', *args, **kwargs):
        self.__model_loader = _ModelLoader(model, *args, *kwargs)

    def __call__(self, img: Union[str, type(PIL.Image)], img_type: str, *args, **kwargs) -> torch.Tensor:
        if img_type == 'pil':
            return self.predict(img)
        if img_type == 'path':
            return self.predict_path(img)
        if img_type == 'url':
            return self.predict_url(img)

    def predict(self, img: type(PIL.Image)) -> torch.Tensor:
        img = self.__transform()(img).unsqueeze(0)

        with torch.no_grad():
            return self.__predictor()(img)

    def predict_path(self, path: str) -> torch.Tensor:
        return self.predict(PIL.Image.open(path))

    def predict_url(self, url: str) -> torch.Tensor:
        return self.predict(PIL.Image.open(requests.get(url, stream=True).raw))

    def __classifier(self) -> modelViT:
        return self.__model_loader.get()

    def __predictor(self) -> modelViT:
        classifier = self.__classifier()
        classifier.fc = _Identity()
        return classifier

    @staticmethod
    def __transform() -> transformCompose:
        return transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ])

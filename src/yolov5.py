import os
import torch
import pandas as pd

from src.config import Configs
from src.datatypes import PILImage, Coordinates


class _ModelLoader:
    __rep_owner = 'ultralytics'
    __rep_name = 'yolov5'

    def __init__(self, model, *args, **kwargs):
        self.__model_name = model
        self.__model_path = f'{Configs.get("directories.models")}/{self.__rep_name}/'
        self.__model = None

        self.__rep_path = f'{self.__model_path}/{self.__rep_owner}_{self.__rep_name}_master'

        self.__args = args
        self.__kwargs = kwargs

    def get(self):
        if not self.__downloaded():
            self.__model = self.__download()

        if self.__model is None:
            self.__model = self.__load()

        return self.__model

    def __load(self):
        return torch.hub.load(self.__rep_path, self.__model_name, source='local', *self.__args, **self.__kwargs)

    def __download(self):
        torch.hub.set_dir(self.__model_path)
        return torch.hub.load(
            f'{self.__rep_owner}/{self.__rep_name}',
            self.__model_name,
            *self.__args,
            **self.__kwargs
        )

    def __downloaded(self):
        return os.path.exists(self.__rep_path) and len(os.listdir(self.__rep_path))


class ModelYoloV5:
    def __init__(self, model: str = 'yolov5s', *args, **kwargs):
        self.__model_loader = _ModelLoader(model, *args, **kwargs)

    def segment(self, img: PILImage) -> list[Coordinates]:
        """ Predict """

        """ get segments """
        segments = self.__model_loader.get()(img)
        """ get segment coordinates """
        coordinates = segments.pandas().xyxy[0]
        coordinates = self.__get_coordinates(coordinates)

        return coordinates

    @staticmethod
    def __get_coordinates(segments: pd.DataFrame) -> list[Coordinates]:
        coordinates = []

        for i in range(len(segments)):
            coordinates.append(Coordinates(
                x_min=segments.xmin[i],
                y_min=segments.ymin[i],
                x_max=segments.xmax[i],
                y_max=segments.ymax[i]
            ))

        return coordinates


""" TODO: we need configs transfer to model """

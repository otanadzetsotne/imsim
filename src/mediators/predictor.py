import torch
from src.modules.predictor import Predictor
from src.dtypes import ImagesInner


class MediatorPredictor:
    @classmethod
    def predict(
            cls,
            model: torch.nn.Module,
            images: ImagesInner,
    ) -> ImagesInner:
        """
        Predict images
        :param model: torch.nn.Module
        :param images: ImagesInner
        :return: ImagesInner
        """

        return Predictor.predict(model, images)

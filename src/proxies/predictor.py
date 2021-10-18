from torch import nn
from src.utils.predictor import Predictor
from src.dtypes import ImagesInner


class ProxyPredictor:
    @classmethod
    def predict(
            cls,
            model: nn.Module,
            images: ImagesInner,
    ) -> ImagesInner:
        """
        Predict images
        :param model: torch.nn.Module
        :param images: ImagesInner
        :return: ImagesInner
        """

        return Predictor.predict(model, images)

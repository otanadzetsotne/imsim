# imported
from torch import nn
# local
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
        """

        return Predictor.predict(model, images)

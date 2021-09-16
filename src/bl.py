from src.facades.mediators import Mediators
from src.dtypes import (
    PredictionInMulti,
    PredictionOutMulti,
    ImagesInner,
    Model,
)


class BusinessLogic:
    mediators = Mediators

    @classmethod
    def predict(
            cls,
            request: PredictionInMulti,
    ):
        """
        Predict images
        """

        images = cls.mediators.downloader.map(request.images)
        images = cls.__predict(request.model, images)

        return PredictionOutMulti(model=request.model, images=images)

    @classmethod
    def __predict(
            cls,
            model: Model,
            images: ImagesInner,
    ):
        """
        Get predicted images
        :param model: model type
        :param images: images to predict
        :return: predicted images
        """

        if cls.mediators.images.has_correct(images):
            model = cls.mediators.collector.collect(model)
            images = cls.mediators.predictor.predict(model, images)

        return images

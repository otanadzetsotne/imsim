from src.facades.mediators import Mediators
from src.dtypes import (
    PredictionInMulti,
    PredictionOutMulti,
    ImagesInner,
    ImagesOut,
    ImageOut,
    Model,
)


class BusinessLogic:
    mediators = Mediators

    @classmethod
    def request_predict(
            cls,
            request: PredictionInMulti,
    ):
        """
        Predict images
        """

        images = cls.mediators.downloader.map(request.images)
        images = cls.predict(request.model, images)
        images = cls.images_inner_to_out(images)

        return PredictionOutMulti(model=request.model, images=images)

    @staticmethod
    def images_inner_to_out(
            images: ImagesInner,
    ) -> ImagesOut:
        """
        Convert ImagesInner object to ImagesOut object
        :param images: images list to convert
        :return: converted images
        """

        return [ImageOut(**dict(image)) for image in images]

    @classmethod
    def predict(
            cls,
            model: Model,
            images: ImagesInner,
    ) -> ImagesInner:
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

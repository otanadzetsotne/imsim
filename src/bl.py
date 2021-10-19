from src.facades.proxies import Proxies
from src.dtypes import (
    PredictionInMulti,
    PredictionOutMulti,
    ImagesInner,
    ImagesOut,
    ImageOut,
    Model,
)


class BusinessLogic:
    proxies = Proxies

    @classmethod
    def request_predict(
            cls,
            request: PredictionInMulti,
    ):
        """
        Predict images
        """

        images = cls.proxies.downloader.map(request.images)
        images = cls.__predict(request.model, images)
        images = cls.__images_inner_to_out(images)

        return PredictionOutMulti(model=request.model, images=images)

    @staticmethod
    def __images_inner_to_out(
            images: ImagesInner,
    ) -> ImagesOut:
        """
        Convert ImagesInner object to ImagesOut object
        :param images: images list to convert
        :return: converted images
        """

        return [ImageOut(**dict(image)) for image in images]

    @classmethod
    def __predict(
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

        if cls.proxies.images.has_correct(images):
            # Load model
            model = cls.proxies.collector.collect(model)

            # Split images
            images_err = cls.proxies.images.filter_error(images)
            images_correct = cls.proxies.images.filter_correct(images)

            # Predict correct ones
            images_correct = cls.proxies.predictor.predict(model, images_correct)

            # Merge images
            images = images_correct + images_err

        return images

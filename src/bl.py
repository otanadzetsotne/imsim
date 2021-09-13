from src.facades.mediators import Mediators
from src.dtypes import (
    AddIn,
    AddOut,
    SearchIn,
    ExistsIn,
    DeleteIn,
    ImagesInner,
    Model,
)


class BusinessLogic:
    mediators = Mediators

    @classmethod
    def add(
            cls,
            request: AddIn,
    ) -> AddOut:
        images = cls.mediators.downloader.map(request.images)
        images = cls.__predict(request.model, images)

        if cls.mediators.images.has_correct(images):
            # TODO: Add to DB
            pass

        return AddOut(model=request.model, images=images)

    @classmethod
    def search(
            cls,
            request: SearchIn,
    ):
        images = [cls.mediators.downloader.one(request.image)]
        images = cls.__predict(request.model, images)

        if cls.mediators.images.has_correct(images):
            # TODO: Search in DB
            pass

        pass

    @classmethod
    def exists(
            cls,
            request: ExistsIn,
    ):
        images = cls.mediators.downloader.one(request.images)
        images = cls.__predict(request.model, images)

        if cls.mediators.images.has_correct(images):
            # TODO: Search in DB
            pass

        pass

    @classmethod
    def delete(
            cls,
            request: DeleteIn,
    ):
        images = cls.mediators.downloader.map(request.images)
        images = cls.__predict(request.model, images)

        if cls.mediators.images.has_correct(images):
            # TODO: Delete from DB
            pass

        pass

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

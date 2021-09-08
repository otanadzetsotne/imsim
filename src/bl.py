from src.mediator import MediatorFacade
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
    mediator = MediatorFacade

    @classmethod
    def add(
            cls,
            request: AddIn,
    ) -> AddOut:
        images = cls.mediator.downloader.map(request.images)
        images = cls.__predict(request.model, images)

        if cls.mediator.images.has_correct(images):
            # TODO: Add to DB
            pass

        return AddOut(model=request.model, images=images)

    @classmethod
    def search(
            cls,
            request: SearchIn,
    ):
        images = [cls.mediator.downloader.one(request.image)]
        images = cls.__predict(request.model, images)

        if cls.mediator.images.has_correct(images):
            # TODO: Search in DB
            pass

        pass

    @classmethod
    def exists(
            cls,
            request: ExistsIn,
    ):
        images = cls.mediator.downloader.one(request.images)
        images = cls.__predict(request.model, images)

        if cls.mediator.images.has_correct(images):
            # TODO: Search in DB
            pass

        pass

    @classmethod
    def delete(
            cls,
            request: DeleteIn,
    ):
        images = cls.mediator.downloader.map(request.images)
        images = cls.__predict(request.model, images)

        if cls.mediator.images.has_correct(images):
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

        if cls.mediator.images.has_correct(images):
            model = cls.mediator.collector.collect(model)
            images = cls.mediator.predictor.predict(model, images)

        return images

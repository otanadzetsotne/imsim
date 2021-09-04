from src.mediator import MediatorFacade
from src.dtypes import (
    PredictionIn,
    PredictionInMulti,
    ImagesIn,
    ImagesInner,
)


class BusinessLogic:
    mediator = MediatorFacade

    @classmethod
    def add(
            cls,
            request: PredictionInMulti,
    ):
        images = cls.mediator.downloader.map(request.images)

        if not cls.mediator.images.has_correct(images):
            # All images with error
            pass

        model = cls.mediator.model.colect(request.model)
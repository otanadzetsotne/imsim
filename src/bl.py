from src.mediator import MediatorFacade
from src.dtypes import (
    AddIn,
    AddOut,
    SearchIn,
    ExistsIn,
    DeleteIn,
)


class BusinessLogic:
    mediator = MediatorFacade

    @classmethod
    def add(
            cls,
            request: AddIn,
    ) -> AddOut:
        images = cls.mediator.downloader.map(request.images)

        if cls.mediator.images.has_correct(images):
            model = cls.mediator.collector.collect(request.model)
            images = cls.mediator.predictor.predict(model, images)

        return AddOut(model=request.model, images=images)

    @classmethod
    def search(
            cls,
            request: SearchIn,
    ):
        image = cls.mediator.downloader.one(request.image)
        # TODO

    @classmethod
    def exists(
            cls,
            request: ExistsIn,
    ):
        image = cls.mediator.downloader.one(request.image)
        # TODO

    @classmethod
    def delete(
            cls,
            request: DeleteIn,
    ):
        # TODO
        pass

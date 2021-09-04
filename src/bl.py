from src.mediator import MediatorFacade
from src.dtypes import (
    AddIn,
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
    ):
        images = cls.mediator.downloader.map(request.images)

        if cls.mediator.images.has_correct(images):
            model = cls.mediator.model.colect(request.model)

        # TODO

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

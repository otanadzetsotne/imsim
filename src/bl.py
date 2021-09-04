from src.modules.downloader import Downloader
from src.dtypes import PredictionIn
from src.dtypes import ImagesIn
from src.dtypes import ImagesInner


class BusinessLogicMediator:
    """
    Mediator class that specifies high level contracts for functions given / received data types
    """

    @staticmethod
    def download_map(
            images: ImagesIn,
    ) -> ImagesInner:
        """
        Downloading images

        :param images: ImagesIn
        :return: ImagesInner
        """

        return Downloader.download_map(images)


class BusinessLogic:
    mediator = BusinessLogicMediator

    @classmethod
    def predict(
            cls,
            prediction_request: PredictionIn,
    ):
        model = prediction_request.model
        image = prediction_request.image

        # get/validate image
        # get model
        # create prediction

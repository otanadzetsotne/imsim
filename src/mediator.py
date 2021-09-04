from src.modules.downloader import Downloader
from src.modules.images import ImagesHelper

from src.dtypes import (
    PredictionIn,
    PredictionInMulti,
    ImagesIn,
    ImagesInner,
)


class MediatorDownloader:
    @staticmethod
    def map(
            images: ImagesIn,
    ) -> ImagesInner:
        """
        Downloading images

        :param images: ImagesIn
        :return: ImagesInner
        """

        return Downloader.map(images)


class MediatorImages:
    @staticmethod
    def filter_correct(
            images: ImagesInner,
    ) -> ImagesInner:
        """
        Filter ImagesInner objects
        :param images: ImagesInner
        :return: ImagesInner
        """

        return ImagesHelper.filter_correct(images)

    @staticmethod
    def filter_error(
            images: ImagesInner,
    ) -> ImagesInner:
        """
        Filter ImagesInner object
        :param images: ImagesInner
        :return: ImagesInner
        """

        return ImagesHelper.filter_error(images)

    @staticmethod
    def has_correct(
            images: ImagesInner,
    ) -> bool:
        """
        Check ImagesInner object
        :param images: ImagesInner
        :return: bool
        """

        return ImagesHelper.has_correct(images)


class MediatorModel:
    pass


class MediatorFacade:
    """
    Mediator facade class that specifies high level contracts for functions given / received data types
    """

    downloader = MediatorDownloader
    images = MediatorImages
    model = MediatorModel

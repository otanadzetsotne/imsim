from src.utils.images import ImagesHelper
from src.dtypes import ImagesInner


class ProxyImages:
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

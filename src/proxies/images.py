from src.utils.images import ImagesHelper
from src.dtypes import ImagesInner


class ProxyImages:
    @staticmethod
    def filter_correct(
            images: ImagesInner,
    ) -> ImagesInner:
        """
        Filter images and return just corrects
        :param images: ImagesInner
        :return: ImagesInner
        """

        return ImagesHelper.filter_correct(images)

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

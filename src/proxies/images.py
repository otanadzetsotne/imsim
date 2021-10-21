from src.utils.images import ImagesHelper
from src.dtypes import ImagesInner


class ProxyImages:
    @staticmethod
    def filter_correct(
            images: ImagesInner,
    ) -> ImagesInner:
        """
        Filter images and return just corrects
        """

        return ImagesHelper.filter_correct(images)

    @staticmethod
    def filter_error(
            images: ImagesInner,
    ) -> ImagesInner:
        """
        Filter images and return just with errors
        """

        return ImagesHelper.filter_error(images)

    @staticmethod
    def has_correct(
            images: ImagesInner,
    ) -> bool:
        """
        Check ImagesInner object
        """

        return ImagesHelper.has_correct(images)

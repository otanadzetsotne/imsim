# local
from src.dtypes import ImagesInner
from config import IMAGE_ERR_CODE_OK


class ImagesHelper:
    @staticmethod
    def filter_correct(
            images: ImagesInner,
    ) -> ImagesInner:
        """
        Filter images and return just corrects
        :param images: images to filter
        :return: images without errors
        """

        return [image for image in images if image.err.code == IMAGE_ERR_CODE_OK]

    @staticmethod
    def filter_error(
            images: ImagesInner,
    ) -> ImagesInner:
        """
        Filter images and return just with errors
        :param images: ImagesInner
        :return: ImagesInner
        """

        return [image for image in images if image.err.code != IMAGE_ERR_CODE_OK]

    @staticmethod
    def has_correct(
            images: ImagesInner,
    ) -> bool:
        """
        Checks if ImagesInner has al least one correct image
        :param images: images to check
        :return: bool
        """

        return any([image.err.code == IMAGE_ERR_CODE_OK for image in images])

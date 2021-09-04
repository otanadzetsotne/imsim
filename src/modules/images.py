from src.dtypes import ImagesInner


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

        return [image for image in images if image.err is None]

    @staticmethod
    def filter_error(
            images: ImagesInner,
    ) -> ImagesInner:
        """
        Filter images and return just with errors
        :param images: images to filter
        :return: images with errors
        """

        return [image for image in images if image.err is not None]

    @staticmethod
    def has_correct(
            images: ImagesInner,
    ) -> bool:
        """
        Checks if ImagesInner has al least one correct image
        :param images: images to check
        :return: bool
        """

        return any([image.err for image in images])

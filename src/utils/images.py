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
    def has_correct(
            images: ImagesInner,
    ) -> bool:
        """
        Checks if ImagesInner has al least one correct image
        :param images: images to check
        :return: bool
        """

        return any([image.err is None for image in images])

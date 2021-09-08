from src.modules.downloader import Downloader
from src.dtypes import ImageIn
from src.dtypes import ImagesIn
from src.dtypes import ImagesInner
from src.dtypes import ImageInner


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

    @staticmethod
    def one(
            image: ImageIn,
    ):
        """
        Download image
        :param image: ImageIn
        :return: ImageInner
        """

        return Downloader.one(image)

from src.modules.downloader import Downloader
from src.dtypes import ImagesIn
from src.dtypes import ImagesInner


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

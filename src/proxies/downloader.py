# local
from src.utils.downloader import Downloader
from src.dtypes import ImagesIn
from src.dtypes import ImagesInner


class ProxyDownloader:
    @staticmethod
    def map(
            images: ImagesIn,
    ) -> ImagesInner:
        """
        Downloading images
        """

        return Downloader.map(images)

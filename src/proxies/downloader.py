from src.utils.downloader import Downloader
from src.dtypes import ImagesIn
from src.dtypes import ImagesInner
from src.dtypes import ModelInput


class ProxyDownloader:
    @staticmethod
    def map(
            model_input: ModelInput,
            images: ImagesIn,
    ) -> ImagesInner:
        """
        Downloading images
        """

        return Downloader.map(model_input, images)

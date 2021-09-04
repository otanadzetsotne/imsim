import io
from multiprocessing import Pool

import requests

from config import (
    IMAGE_PIL_FORMAT,
    IMAGE_CONTENT_TYPES,
)
from src.dtypes import (
    ImageIn,
    ImagesIn,
    ImageInner,
    ImagesInner,
    ImagePIL,
)
from src.exceptions import BadUrlError


class Downloader:
    @staticmethod
    def download(
            image: ImageIn,
    ) -> ImageInner:
        """
        Download image
        :param image: request image
        :return: inner representation of image
        """

        image_inner = ImageInner(**dict(image))

        try:
            # Make request
            response = requests.get(image.url, stream=True)

            # Invalid url if response is None
            if response is None or response.headers['Content-Type'] not in IMAGE_CONTENT_TYPES:
                raise BadUrlError

            # Raises exception response code != 200
            response.raise_for_status()

            # Create PIL
            image_response = response.content
            image_bytes = io.BytesIO(image_response)
            image_pil = ImagePIL.open(image_bytes)
            image_pil = image_pil.convert(IMAGE_PIL_FORMAT)

            # Update image object
            image_inner.pil = image_pil

        except Exception as e:
            image_inner.err = e

        return image_inner

    @classmethod
    def download_map(
            cls,
            images: ImagesIn,
    ) -> ImagesInner:
        """
        Images parallel download
        :param images: request images
        :return: inner representation of images
        """

        with Pool() as pool:
            images = pool.map(cls.download, images)

        return ImagesInner(images=images)

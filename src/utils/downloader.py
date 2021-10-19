import io
from multiprocessing import Pool

import requests
from requests.models import HTTPError

from config import (
    IMAGE_PIL_FORMAT,
    IMAGE_CONTENT_TYPES,
    IMAGE_ERR_CODE_OK,
)
from src.dtypes import (
    ImageIn,
    ImagesIn,
    ImageInner,
    ImagesInner,
    ImagePILModule,
    ImageError,
)
from src.exceptions import BadUrlError


class Downloader:
    @staticmethod
    def one(
            image: ImageIn,
    ) -> ImageInner:
        """
        Download image
        :param image: request image
        :return: inner representation of image
        """

        image_inner = ImageInner(
            **dict(image),
            err=ImageError(code=IMAGE_ERR_CODE_OK),
        )

        try:
            # Make request
            response = requests.get(image.url, stream=True)

            # Raises exception response code != 200
            response.raise_for_status()

            # Invalid url if response is None
            if response.headers['Content-Type'] not in IMAGE_CONTENT_TYPES:
                raise BadUrlError

            # Create PIL
            image_response = response.content
            image_bytes = io.BytesIO(image_response)
            image_pil = ImagePILModule.open(image_bytes)
            image_pil = image_pil.convert(IMAGE_PIL_FORMAT)

            # Update image object
            image_inner.pil = image_pil

        # TODO: exceptions logic refactoring
        except BadUrlError as e:
            image_inner.err.code = e.code
            image_inner.err.desc = f'{e.code} Client error: Bad Request for url {image.url}'
        except HTTPError as e:
            image_inner.err.code = e.response.status_code
            image_inner.err.desc = str(e)
        except Exception:
            code = 500
            image_inner.err.code = code
            image_inner.err.desc = f'{code} Server error: Internal Server Error for url {image.url}'

        return image_inner

    @classmethod
    def map(
            cls,
            images: ImagesIn,
    ) -> ImagesInner:
        """
        Images parallel download
        :param images: request images
        :return: inner representation of images
        """

        with Pool() as pool:
            images = pool.map(cls.one, images)

        return images

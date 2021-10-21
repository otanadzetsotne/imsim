import io
from itertools import repeat
from multiprocessing import Pool

import requests
from requests.models import HTTPError

from config import (
    IMAGE_PIL_FORMAT,
    IMAGE_CONTENT_TYPES,
    IMAGE_ERR_CODE_OK,
    IMAGE_PIL_RESAMPLE,
)
from src.dtypes import (
    ImageIn,
    ImagesIn,
    ImageInner,
    ImagesInner,
    ImagePILModule,
    ImageError,
    ModelInput,
)
from src.exceptions import BadUrlError


class Downloader:
    @staticmethod
    def one(
            model_input: ModelInput,
            image: ImageIn,
    ) -> ImageInner:
        """
        Download image

        :param model_input: model input size
        :param image: request image
        :return: inner representation of image
        """

        image_size = model_input.value

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
            image_pil = image_pil.resize((image_size, image_size), IMAGE_PIL_RESAMPLE)
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
    def one_proxy(
            cls,
            args,
    ):
        """
        Proxy function for Pool().map
        Unpacks multiple arguments into cls.one() function

        :param args: Iterable[ModelType, ImageInner]
        """

        return cls.one(*args)

    @classmethod
    def map(
            cls,
            model_input: ModelInput,
            images: ImagesIn,
    ) -> ImagesInner:
        """
        Images parallel download

        :param model_input: model input size
        :param images: request images
        :return: inner representation of images
        """

        with Pool() as pool:
            images = pool.map(
                cls.one_proxy,
                zip(repeat(model_input), images),
            )

        return images

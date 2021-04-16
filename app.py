import grequests
import itertools
from PIL import Image
from requests.models import Response
from concurrent.futures import ThreadPoolExecutor
from src.vit import ModelViT
from src.encoder import ModelEncoder
from src.datatypes import ImageData


def _predict_vit(model: ModelViT, image: ImageData) -> ImageData:
    """ ViT predict """

    if image.err is None:
        try:
            data = model.predict(image.pil)
            image = image._replace(data=data)
        except Exception as err:
            image = image._replace(err=err)

    return image


def _predict_vit_proxy(arguments: tuple[ModelViT, ImageData]) -> ImageData:
    """ ViT predict arguments proxy """

    return _predict_vit(arguments[0], arguments[1])


class Application:
    """ Application class with high level business logic """

    def __init__(self, segmentation: bool):
        self.__vit = ModelViT()
        self.__encoder = ModelEncoder()
        self.__segmentation = segmentation

    def predict_urls(self, urls: list[str]) -> list[ImageData]:
        """ Create predictions for urls """

        responses = self.__request(urls)

        images = self.__create_image_map(urls, responses)
        images = self.__predict_vit_map(images)
        images = self.__predict_encoder_map(images)

        return images

    def __predict_vit_map(self, images: list[ImageData]) -> list[ImageData]:
        """ Images mapping for ViT predict """

        with ThreadPoolExecutor() as executor:
            images = executor.map(_predict_vit_proxy, zip(itertools.repeat(self.__vit), images))
            images = [_ for _ in images]

        return images

    def __predict_encoder_map(self, images: list[ImageData]) -> list[ImageData]:
        """ Images mapping for Encoder predict """

        images = map(self.__predict_encoder, images)
        images = [_ for _ in images]

        return images

    def __predict_encoder(self, image: type(ImageData)) -> ImageData:
        """ Encoder predict """

        if image.err is None:
            try:
                data = self.__encoder.predict(image.data, workers=32, use_multiprocessing=True)
                data = data.reshape(-1)
                image = image._replace(data=data)
            except Exception as err:
                image = image._replace(err=err)

        return image

    def __create_image_map(self, urls: list[str], responses: list[Response]):
        """ Responses mapping for create ImageData """

        images = map(self.__create_image, urls, responses)
        images = [_ for _ in images]

        return images

    @staticmethod
    def __create_image(url: str, response: Response) -> ImageData:
        """ Create ImageData """

        image = ImageData(url=url)

        try:
            """ raise Exception if response is None """
            if response is None:
                raise Exception('Incorrect Url')

            """ if the response was successful, no Exception will be raised """
            response.raise_for_status()

            """ save PIL.Image object """
            image = image._replace(pil=Image.open(response.raw).convert('RGB'))

        except Exception as err:
            """ save error """
            image = image._replace(err=err)

        return image

    @staticmethod
    def __request(urls: list[str]) -> list[Response]:
        """ Request urls list """

        """ create requests list for each url """
        requests_list = [grequests.get(url) for url in urls]
        """ async requests to requests list """
        responses = grequests.map(requests_list, stream=True, size=1000)

        return responses

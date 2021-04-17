import grequests
from PIL import Image
from typing import Iterable
from requests.models import Response

from src.datatypes import ImageData, ImageSegment, Coordinates
from src.predictors import PredictorVit, PredictorEncoder, PredictorYoloV5


class Application:
    """ Application class with high level business logic """

    __predictor_vit = PredictorVit
    __predictor_encoder = PredictorEncoder
    __predictor_yolov5 = PredictorYoloV5

    def __init__(self, segmentation: bool):
        self.__segmentation = segmentation

    def predict_urls(self, urls: Iterable[str]) -> Iterable[ImageData]:
        """ Create predictions for urls """

        responses = self.__request(urls)
        images = self.__create_image_map(urls, responses)
        images = self.predict(images)

        return images

    def predict(self, images: Iterable[ImageData]) -> Iterable[ImageData]:
        if self.__segmentation:
            images = self.__predictor_yolov5.predict(images)

        images = self.__predictor_vit.predict(images)
        images = self.__predictor_encoder.predict(images)

        return images

    def __create_image_map(self, urls: Iterable[str], responses: Iterable[Response]):
        """ Responses mapping for create ImageData """

        return map(self.__create_image, urls, responses)

    @staticmethod
    def __create_image(url: str, response: Response) -> ImageData:
        """ Create ImageData """

        image = ImageData(path=url)

        try:
            """ raise Exception if response is None """
            if response is None:
                raise Exception('Bad url')

            """ if the response was successful, no Exception will be raised """
            response.raise_for_status()

            """ create ImageSegment """
            pil = Image.open(response.raw).convert('RGB')
            coordinates = Coordinates(x_min=0, y_min=0, x_max=pil.size[0], y_max=pil.size[1])
            segment = ImageSegment(pil=pil, coordinates=coordinates, is_full=True)

            """ save ImageSegment to ImageData """
            image = image.add_segment(segment)

        except Exception as err:
            """ save error """
            image = image._replace(err=err)

        return image

    @staticmethod
    def __request(urls: Iterable[str]) -> Iterable[Response]:
        """ Request urls list """

        """ create requests list for each url """
        requests_list = [grequests.get(url) for url in urls]
        """ async requests to requests list """
        responses = grequests.map(requests_list, stream=True, size=1000)

        return responses

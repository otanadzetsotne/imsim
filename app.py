import grequests
from typing import Iterable
from requests.models import Response

from src.datatypes import ImageData
from src.datahelpers import ImageDataCreator
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
        """ Create predictions """

        if self.__segmentation:
            images = self.__predictor_yolov5.predict(images)

        images = self.__predictor_vit.predict(images)
        images = self.__predictor_encoder.predict(images)

        return images

    @staticmethod
    def __create_image_map(urls: Iterable[str], responses: Iterable[Response]) -> Iterable[ImageData]:
        """ Responses mapping for create ImageData """

        return map(ImageDataCreator.create_by_response, urls, responses)

    @staticmethod
    def __request(urls: Iterable[str]) -> Iterable[Response]:
        """ Request urls list """

        """ create requests list for each url """
        requests_list = [grequests.get(url) for url in urls]
        """ async requests to requests list """
        responses = grequests.map(requests_list, stream=True, size=1000)

        return responses

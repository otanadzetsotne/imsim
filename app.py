from typing import Iterable
from time import perf_counter

import grequests
from requests.models import Response

from src.timer import timer
from src.datatypes import ImageData
from src.datahelpers import ImageDataCreator
from src.predictors import PredictorVit, PredictorEncoder


class Application:
    def __init__(
            self,
    ):
        self.__predictor_encoder = PredictorEncoder()
        self.__predictor_vit = PredictorVit()

    """ Application class with high level business logic """

    def predict_urls(self, urls: Iterable[str]) -> Iterable[ImageData]:
        """ Create predictions for urls """

        responses = self.__requests(urls)
        images = self.__create_image_map(urls, responses)
        images = self.predict(images)

        return images

    def predict(self, images: Iterable[ImageData]) -> Iterable[ImageData]:
        """ Create predictions """

        images = self.__predict_vit(images)
        images = self.__predict_encoder(images)

        return images

    @timer
    def __predict_vit(self, images: Iterable[ImageData]) -> Iterable[ImageData]:
        """ Get VIT predictions """

        return list(self.__predictor_vit.predict(images))

    @timer
    def __predict_encoder(self, images: Iterable[ImageData]) -> Iterable[ImageData]:
        """ Get encoder predictions """

        return list(self.__predictor_encoder.predict(images))

    @staticmethod
    def __create_image_map(urls: Iterable[str], responses: Iterable[Response]) -> Iterable[ImageData]:
        """ Responses mapping for create ImageData """

        return map(ImageDataCreator.create_by_response, urls, responses)

    @staticmethod
    @timer
    def __requests(urls: Iterable[str]) -> Iterable[Response]:
        """ Request urls list """

        """ create requests list for each url """
        requests_list = [grequests.get(url, stream=True, verify=False) for url in urls]
        """ async requests to requests list """
        responses = grequests.map(requests_list, size=1000)

        return responses

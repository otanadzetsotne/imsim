from typing import Iterable
from abc import ABCMeta, abstractmethod
from concurrent.futures import ThreadPoolExecutor

from src.vit import ModelViT
from src.encoder import ModelEncoder
from src.datatypes import ImageData, ImageSegment


class PredictorEncoder:
    __model = ModelEncoder()

    def predict(self, images: Iterable[ImageData]) -> Iterable[ImageData]:
        return map(self.__predict_image, images)

    def __predict_image(self, image: ImageData) -> ImageData:
        if image.err is None:
            try:
                # segments = [self.__predict_segment(seg) for seg in image.segments]
                segments = list(map(self.__predict_segment, image.segments))
                image = image._replace(segments=segments)
            except Exception as err:
                image = image._replace(err=err)

        return image

    def __predict_segment(self, segment: ImageSegment) -> ImageSegment:
        if segment.err is None:
            try:
                # data = self.__model.predict(segment.data)
                segment = segment._replace(data=self.__model.predict(segment.data.reshape(-1)))
            except Exception as err:
                print(err)
                segment = segment._replace(err=err)

        return segment


class PredictorVit:
    def __init__(self):
        self.__model = ModelViT()

    def predict(self, images: Iterable[ImageData]) -> Iterable[ImageData]:
        with ThreadPoolExecutor() as executor:
            images = executor.map(self.__predict_image, images)

        return images

    def __predict_image(self, image: ImageData) -> ImageData:
        if image.err is None:
            try:
                segments = list(map(self.__predict_segment, image.segments))
                image = image._replace(segments=segments)
            except Exception as err:
                image = image._replace(err=err)

        return image

    def __predict_segment(self, segment: ImageSegment) -> ImageSegment:
        if segment.err is None:
            try:
                segment = segment._replace(data=self.__model.predict(segment.pil))
            except Exception as err:
                segment = segment._replace(err=err)

        return segment

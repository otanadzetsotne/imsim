from typing import Iterable
from abc import ABCMeta, abstractmethod
from concurrent.futures import ThreadPoolExecutor

from src.vit import ModelViT
from src.encoder import ModelEncoder
from src.yolov5 import ModelYoloV5
from src.datatypes import ImageData, ImageSegment


class PredictorAbstract(metaclass=ABCMeta):
    __model = None

    @classmethod
    @abstractmethod
    def predict(cls, images: Iterable[ImageData]) -> Iterable[ImageData]:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def __predict_image(cls, image: ImageData) -> ImageData:
        raise NotImplementedError()


class PredictorYoloV5(PredictorAbstract):
    __model = ModelYoloV5()

    @classmethod
    def predict(cls, images: Iterable[ImageData], processes: int = 1) -> Iterable[ImageData]:
        with ThreadPoolExecutor(processes) as executor:
            images = executor.map(cls.__predict_image, images)

        return images

    @classmethod
    def __predict_image(cls, image: ImageData) -> ImageData:
        if image.err is None:
            """ get full segment of entire image """
            full_segment = image.get_segment_full()
            """ get all defined objects coordinates """
            coordinates_all = cls.__model.segment(full_segment.pil)

            for coordinates in coordinates_all:
                try:
                    """ create segment data """
                    crop_coordinate = coordinates.get_for_crop_pil()
                    pil = full_segment.pil.crop(crop_coordinate)

                    """ add new segment """
                    image = image.add_segment(ImageSegment(pil=pil, coordinates=coordinates))
                finally:
                    """ if something went wrong we just continue """
                    continue
            pass

        return image


class PredictorEncoder(PredictorAbstract):
    __model = ModelEncoder()

    @classmethod
    def predict(cls, images: Iterable[ImageData], processes: int = 1) -> Iterable[ImageData]:
        with ThreadPoolExecutor(processes) as executor:
            images = executor.map(cls.__predict_image, images)

        return images

    @classmethod
    def __predict_image(cls, image: ImageData) -> ImageData:
        if image.err is None:
            try:
                segments = list(map(cls.__predict_segment, image.segments))
                image = image._replace(segments=segments)
            except Exception as err:
                image = image._replace(err=err)

        return image

    @classmethod
    def __predict_segment(cls, segment: ImageSegment) -> ImageSegment:
        if segment.err is None:
            try:
                segment = segment._replace(data=cls.__model.predict(segment.data))
            except Exception as err:
                segment = segment._replace(err=err)

        return segment


class PredictorVit(PredictorAbstract):
    __model = ModelViT()

    @classmethod
    def predict(cls, images: Iterable[ImageData], processes: int = 1) -> Iterable[ImageData]:
        with ThreadPoolExecutor(processes) as executor:
            images = executor.map(cls.__predict_image, images)

        return images

    @classmethod
    def __predict_image(cls, image: ImageData) -> ImageData:
        if image.err is None:
            try:
                segments = list(map(cls.__predict_segment, image.segments))
                image = image._replace(segments=segments)
            except Exception as err:
                image = image._replace(err=err)

        return image

    @classmethod
    def __predict_segment(cls, segment: ImageSegment) -> ImageSegment:
        if segment.err is None:
            try:
                segment = segment._replace(data=cls.__model.predict(segment.pil))
            except Exception as err:
                segment = segment._replace(err=err)

        return segment

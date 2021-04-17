import grequests
import itertools
from PIL import Image
from requests.models import Response
from concurrent.futures import ThreadPoolExecutor
from src.vit import ModelViT
from src.encoder import ModelEncoder
from src.yolov5 import ModelYoloV5
from src.datatypes import ImageData, ImageSegment, Coordinates


def _predict_vit_segment(model: ModelViT, segment: ImageSegment):
    """ ViT predict """

    if segment.err is None:
        try:
            segment = segment._replace(data=model.predict(segment.pil))
        except Exception as err:
            segment = segment._replace(err=err)

    return segment


def _predict_vit(arguments: tuple[ModelViT, ImageData]) -> ImageData:
    """ ViT predict segments """

    vit = arguments[0]
    image = arguments[1]

    if image.err is None:
        segments = list(map(_predict_vit_segment, itertools.repeat(vit), image.segments))
        image = image._replace(segments=segments)

    return image


class Application:
    """ Application class with high level business logic """

    def __init__(self, segmentation: bool):
        self.__vit = ModelViT()
        self.__encoder = ModelEncoder()
        self.__yolov5 = ModelYoloV5()
        self.__segmentation = segmentation

    def predict_urls(self, urls: list[str]) -> list[ImageData]:
        """ Create predictions for urls """

        responses = self.__request(urls)

        images = self.__create_image_map(urls, responses)
        images = self.predict(images)

        return images

    def predict(self, images: list[ImageData]) -> list[ImageData]:
        if self.__segmentation:
            images = self.__predict_segmentation_map(images)

        images = self.__predict_vit_map(images)
        images = self.__predict_encoder_map(images)

        return images

    def __predict_segmentation_map(self, images: list[ImageData]):
        images = map(self.__predict_segmentation, images)
        images = [_ for _ in images]

        return images

    def __predict_segmentation(self, image: ImageData):
        if image.err is None:
            """ get full segment of entire image """
            full_segment = image.get_segment_full()
            """ get all defined objects coordinates """
            coordinates_all = self.__yolov5.segment(full_segment.pil)

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

        return image

    def __predict_vit_map(self, images: list[ImageData]) -> list[ImageData]:
        """ Images mapping for ViT predict """

        with ThreadPoolExecutor() as executor:
            images = executor.map(_predict_vit, zip(itertools.repeat(self.__vit), images))
            images = [_ for _ in images]

        return images

    def __predict_encoder_map(self, images: list[ImageData]) -> list[ImageData]:
        """ Images mapping for Encoder predict """

        images = map(self.__predict_encoder, images)
        images = [_ for _ in images]

        return images

    def __predict_encoder(self, image: ImageData) -> ImageData:
        """ Encoder predict """

        if image.err is None:
            try:
                segments = list(map(self.__predict_encoder_segment, image.segments))
                image = image._replace(segments=segments)
            except Exception as err:
                image = image._replace(err=err)

        return image

    def __predict_encoder_segment(self, segment: ImageSegment) -> ImageSegment:
        if segment.err is None:
            try:
                data = self.__encoder.predict(segment.data, workers=32, use_multiprocessing=True)
                segment = segment._replace(data=data)

            except Exception as err:
                segment = segment._replace(err=err)

        return segment

    def __create_image_map(self, urls: list[str], responses: list[Response]):
        """ Responses mapping for create ImageData """

        images = map(self.__create_image, urls, responses)
        images = [_ for _ in images]

        return images

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
    def __request(urls: list[str]) -> list[Response]:
        """ Request urls list """

        """ create requests list for each url """
        requests_list = [grequests.get(url) for url in urls]
        """ async requests to requests list """
        responses = grequests.map(requests_list, stream=True, size=1000)

        return responses

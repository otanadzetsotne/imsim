from PIL import Image
from requests.models import Response

from src.datatypes import ImageData, ImageSegment, Coordinates


class ImageDataCreator:
    @staticmethod
    def create_by_response(url: str, response: Response) -> ImageData:
        """ Hides low level ImageData creation logic """

        image = ImageData(path=url)

        try:
            """ raise Exception if response is None """
            if response is None:
                raise Exception(f'Bad url: {url}')

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
    def create_by_path(path: str) -> ImageData:
        """ Hides low level ImageData creation logic """

        image = ImageData(path=path)

        try:
            """ create ImageSegment """
            pil = Image.open(path).convert('RGB')
            coordinates = Coordinates(x_min=0, y_min=0, x_max=pil.size[0], y_max=pil.size[1])
            segment = ImageSegment(pil=pil, coordinates=coordinates, is_full=True)

            """ save ImageSegment to ImageData """
            image = image.add_segment(segment)

        except Exception as err:
            """ save error """
            image = image._replace(err=err)

        return image


class ImageDataHelper:
    @staticmethod
    def prune_segments(image: ImageData) -> ImageData:
        segment_full = image.get_segment_full()
        center_full = segment_full.coordinates.get_center()
        allowable_distance = (center_full[0] // 2, center_full[1] // 2)

        segments_pruned = []
        for segment in image.segments:
            center = segment.coordinates.get_center()

            is_valid = abs(center_full[0] - center[0]) < allowable_distance[0]
            is_valid = abs(center_full[1] - center[1]) < allowable_distance[1] and is_valid

            if is_valid:
                segments_pruned.append(segment)

        return image._replace(segments=segments_pruned)

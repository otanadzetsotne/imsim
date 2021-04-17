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

import numpy as np
from PIL import Image
from typing import Optional, NamedTuple


PILImage = type(Image)


class CoordinatesDot(NamedTuple):
    x: float
    y: float

    def __repr__(self):
        output = ''
        output += f'x: {self.x}, '
        output += f'y: {self.y}, '

        return output


class Coordinates(NamedTuple):
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __repr__(self):
        output = ''
        output += f'x_min: {self.x_min}, '
        output += f'y_min: {self.y_min}, '
        output += f'x_max: {self.x_max}, '
        output += f'y_max: {self.y_max}, '

        return output

    def get_for_crop_pil(self) -> tuple[float, float, float, float]:
        return self.x_min, self.y_min, self.x_max, self.y_max

    def get_center(self):
        center_x = (self.x_min + self.x_max) / 2
        center_y = (self.y_min + self.y_max) / 2

        return center_x, center_y


class ImageSegment(NamedTuple):
    pil: Optional[PILImage]
    coordinates: Optional[Coordinates]
    data: Optional[np.ndarray] = None
    err: Optional[Exception] = None
    is_full: bool = False

    def __repr__(self):
        return f'{self.__class__} with coordinates: ({self.coordinates})'


class ImageData(NamedTuple):
    """ NamedTuple based class for image data transfer """

    path: str
    segments: Optional[list[ImageSegment]] = []
    err: Optional[Exception] = None

    def __repr__(self):
        output = ''
        output += f'{self.__class__}\n'
        output += f'path: {self.path}\n'
        output += f'segments: {len(self.segments)}\n'
        output += f'err: {self.err.__repr__()}'

        return output

    def add_segment(self, segment: ImageSegment):
        """ Add new segment to ImageData segments """

        segments = self.segments + [segment]
        new = self._replace(segments=segments)

        return new

    def get_segment_full(self) -> Optional[ImageSegment]:
        """ Get segment of entire image """

        return [segment for segment in self.segments if segment.is_full][0]

    @staticmethod
    def __calculate_distance(segment_a: ImageSegment, segment_b: ImageSegment) -> float:
        """ Calculate distance between segments """
        """ TODO: this should be in another place """
        
        """ get center coordinates """
        center_a = segment_a.coordinates.get_center()
        center_b = segment_b.coordinates.get_center()
        
        """ calculate distance between center coordinates """
        distance = (((center_a[0] - center_b[0]) ** 2) + ((center_a[1] - center_b[1]) ** 2)) ** .5

        return distance

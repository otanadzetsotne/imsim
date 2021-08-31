from statistics import mean, harmonic_mean

import numpy as np
from PIL import Image
from typing import Optional, NamedTuple


PILImage = type(Image)


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
        output = ''
        output += f'{self.__class__}\n'
        output += f'coordinates: {self.coordinates}\n'
        output += f'data: {self.data.shape}\n' if self.data is not None else f'err: {self.err}'
        output += f'is_full: {self.is_full}'
        return output


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
        output += f'err: {self.err}\n'

        return output

    def add_segment(self, segment: ImageSegment):
        """ Add new segment to ImageData segments """

        segments = self.segments + [segment]
        new = self._replace(segments=segments)

        return new

    def get_segment_full(self) -> Optional[ImageSegment]:
        """ Get segment of entire image """

        return [segment for segment in self.segments if segment.is_full][0]

    def __get_data(self) -> Optional[np.ndarray]:
        # create segments data matrix
        data = np.array([segment.data for segment in self.segments])
        # transpose matrix for geometric_mean()
        data = np.transpose(data)

        return data

    def get_data_mean(self) -> Optional[np.ndarray]:
        """ Get segments data vector """

        data = self.__get_data()
        # create geometric mean vector
        data = np.array([mean(vector.reshape(-1)) for vector in data])

        return data

    def get_data_mean_harmonic(self) -> Optional[np.ndarray]:
        """ Get segments data vector """

        data = self.__get_data()
        # create geometric mean vector
        data = np.array([harmonic_mean(vector) for vector in data])

        return data

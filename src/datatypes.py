import numpy as np
from PIL import Image
from typing import Optional, NamedTuple


class ImageData(NamedTuple):
    """ NamedTuple based class for image data transfer """

    url: str
    pil: Optional[type(Image)] = None
    data: Optional[np.ndarray] = None
    err: Optional[Exception] = None

    def __repr__(self):
        output = ''
        output += f'{self.__class__}\n'
        output += f'url: {self.url}\n'
        output += f'pil: {self.pil}\n'
        output += f'data: {self.data}\n'
        output += f'err: {self.err.__repr__()}\n'

        return output

# standard
from enum import Enum
from typing import Optional, Union
# imported
from torch import Tensor
from pydantic import BaseModel
from pydantic import HttpUrl
from PIL.Image import Image as ImagePIL
from PIL import Image as ImagePILModule


# Neural


class Model(Enum):
    vit: str = 'vit'
    test: str = 'test'


# Image pydantic models


class ImageError(BaseModel):
    code: int
    desc: Optional[str] = None


class Image(BaseModel):
    url: HttpUrl


class ImagePredicted(Image):
    prediction: Union[Tensor, list[float]] = None
    err: ImageError

    class Config:
        arbitrary_types_allowed = True


class ImageInner(ImagePredicted):
    pil: Optional[ImagePIL] = None


class ImageIn(Image):
    pass


class ImageOut(ImagePredicted):
    pass


ImagesIn = list[ImageIn]
ImagesInner = list[ImageInner]
ImagesOut = list[ImageOut]


# Prediction pydantic models


class Prediction(BaseModel):
    model: Model


class PredictionInMulti(Prediction):
    images: ImagesIn


class PredictionOutMulti(Prediction):
    images: ImagesOut

from enum import Enum
from typing import List
from typing import Optional

from PIL import Image as ImagePIL
from pydantic import BaseModel
from pydantic import HttpUrl


# Neural


class Model(Enum):
    vit = 'vit'


# Image pydantic models


ImagePILType = type(ImagePIL)


class Image(BaseModel):
    url: Optional[HttpUrl] = None


class ImagePredicted(Image):
    prediction: Optional[List[float]] = None
    err: Optional[type(Exception)] = None


class ImageInner(ImagePredicted):
    pil: Optional[ImagePILType] = None


class ImageIn(Image):
    pass


class ImageOut(ImagePredicted):
    pass


ImagesIn = List[ImageIn]
ImagesInner = List[ImageInner]
ImagesOut = List[ImageOut]


# Prediction pydantic models


class Prediction(BaseModel):
    model: Model


class PredictionInMulti(Prediction):
    images: ImagesIn


class PredictionOutMulti(Prediction):
    images: ImagesOut

from enum import Enum
from typing import List
from typing import Optional

from PIL import Image as ImagePIL
from pydantic import BaseModel
from pydantic import HttpUrl

from config import EXISTS_SCORE


# Neural


class Model(Enum):
    vit = 'vit'


# Image pydantic models


ImagePILType = type(ImagePIL)


class Image(BaseModel):
    url: Optional[HttpUrl] = None
    id_custom: Optional[int] = None

    class Config:
        arbitrary_types_allowed = True


class ImageInner(Image):
    pil: Optional[ImagePILType] = None
    prediction: Optional[List[float]] = None
    err: Optional[type(Exception)] = None


class ImageOut(Image):
    err: Optional[Exception] = None


class ImageIn(Image):
    pass


ImagesIn = List[ImageIn]
ImagesInner = List[ImageInner]
ImagesOut = List[ImageOut]


# Prediction pydantic models


class Prediction(BaseModel):
    model: Model


class PredictionIn(Prediction):
    image: ImageIn


class PredictionInMulti(Prediction):
    images: ImagesIn


class PredictionOutMulti(Prediction):
    images: ImagesOut


# High level requests pydantic models


class AddIn(PredictionInMulti):
    pass


class AddOut(PredictionOutMulti):
    pass


class SearchIn(PredictionIn):
    pass


class DeleteIn(PredictionInMulti):
    pass


class ExistsIn(PredictionIn):
    score: float = EXISTS_SCORE

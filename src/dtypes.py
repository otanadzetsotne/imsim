from enum import Enum
from typing import List
from typing import Optional

from PIL import Image as ImagePIL
from pydantic import BaseModel
from pydantic import HttpUrl

from config import EXISTS_SCORE


# Neural


class Model(Enum):
    vit_l = 'vit_l'
    vit_m = 'vit_m'
    vit_s = 'vit_s'


# Image pydantic models


ImagePILType = type(ImagePIL)


class Image(BaseModel):
    url: Optional[HttpUrl] = None
    client_id: Optional[int] = None

    class Config:
        arbitrary_types_allowed = True


class ImageInner(Image):
    pil: Optional[ImagePILType] = None
    err: Optional[type(Exception)] = None


class ImageOut(Image):
    pass


class ImageIn(Image):
    pass


class ImagesIn(BaseModel):
    images: List[ImageIn]


class ImagesInner(BaseModel):
    images: List[ImageInner]


# Prediction pydantic models


class Prediction(BaseModel):
    model: Model


class PredictionIn(Prediction):
    image: ImageIn


class PredictionInMulti(Prediction):
    images: ImagesIn


# High level requests pydantic models


class AddIn(PredictionInMulti):
    pass


class SearchIn(PredictionIn):
    pass


class DeleteIn(PredictionInMulti):
    pass


class ExistsIn(PredictionIn):
    score: float = EXISTS_SCORE

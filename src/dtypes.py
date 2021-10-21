from enum import Enum
from typing import Optional, Union

from torch import Tensor
from PIL.Image import Image as ImagePIL
from PIL import Image as ImagePILModule
from pydantic import BaseModel
from pydantic import HttpUrl


# Neural


class Model(Enum):
    vit: str = 'vit'


class ModelInput(Enum):
    s: int = 192
    m: int = 480


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
    model_input = ModelInput


class PredictionInMulti(Prediction):
    images: ImagesIn


class PredictionOutMulti(Prediction):
    images: ImagesOut

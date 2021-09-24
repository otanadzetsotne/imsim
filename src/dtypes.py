from enum import Enum
from typing import Optional

import torch
from PIL import Image as ImagePIL
from pydantic import BaseModel
from pydantic import HttpUrl


# Neural


class Model(Enum):
    vit = 'vit'


# Image pydantic models


ImagePILType = type(ImagePIL)


class Image(BaseModel):
    url: HttpUrl


class ImagePredicted(Image):
    prediction: Optional[torch.Tensor] = None
    err: Optional[Exception] = None

    class Config:
        arbitrary_types_allowed = True


class ImageInner(ImagePredicted):
    pil: Optional[ImagePILType] = None


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

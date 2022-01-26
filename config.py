import os
from datetime import timedelta

from PIL import Image
from pydantic import BaseModel, BaseSettings


# Generate with openssl rand -hex 32
SECRET_KEY = '57ddde524dbf423ea7d1c3bba3dc6a24bbca3dbc4e0a0c0d08d9b729f2db8779'
ACCESS_TOKEN_ALGORITHM = 'HS256'
ACCESS_TOKEN_EXPIRES = timedelta(minutes=30)
ACCESS_TOKEN_TYPE = 'bearer'

APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Downloader configs

IMAGE_SIZE = 480
IMAGE_PIL_FORMAT = 'RGB'
IMAGE_PIL_RESAMPLE = Image.BICUBIC
IMAGE_CONTENT_TYPES = [
    'image/gif',
    'image/jpeg',
    'image/png',
    'image/tiff',
    'image/vnd.microsoft.icon',
    'image/x-icon',
    'image/vnd.djvu',
    'image/svg+xml',
]

# Models configs

MODEL_DIR = os.path.join(APP_DIR, 'models')
MODEL_NAME_VIT = 'vit'

# Visual Transformer configs

# Models from https://github.com/lukemelas/PyTorch-Pretrained-ViT
MODEL_VIT_TYPE = 'B_16'

MODEL_VIT_PATH = os.path.join(
    MODEL_DIR,
    f'{MODEL_VIT_TYPE}_{IMAGE_SIZE}.pickle',
)

MODEL_VIT_ENCODER_NAME = 'ViT_encoder'
MODEL_VIT_ENCODER_PATH = os.path.join(MODEL_DIR, f'{MODEL_VIT_ENCODER_NAME}.pickle')

# Data Types configs

IMAGE_ERR_CODE_OK = 200


# New configs


class _SettingsSecret(BaseModel):
    key_token: str  # SECRET_KEY


class _SettingsPaths(BaseModel):
    models: str = 'models'  # MODEL_DIR


class _SettingsViT(BaseModel):
    name: str = 'ViT'  # MODEL_NAME_VIT
    type: str = 'B_16'  # MODEL_VIT_TYPE


class _SettingsToken(BaseModel):
    type: str = 'bearer'  # ACCESS_TOKEN_ALGORITHM
    algorithm: str = 'HS256'  # ACCESS_TOKEN_EXPIRES
    expires: timedelta = timedelta(minutes=30)  # ACCESS_TOKEN_TYPE


class _SettingsImage(BaseModel):
    size: int = 480  # IMAGE_SIZE
    format: str = 'RGB'  # IMAGE_PIL_FORMAT
    resample: int = Image.BICUBIC  # IMAGE_PIL_RESAMPLE
    allowed_types: list[str] = [  # IMAGE_CONTENT_TYPES
        'image/gif',
        'image/jpeg',
        'image/png',
        'image/tiff',
        'image/vnd.microsoft.icon',
        'image/x-icon',
        'image/vnd.djvu',
        'image/svg+xml',
    ]


class Settings(BaseSettings):
    secret: _SettingsSecret
    token: _SettingsToken = _SettingsToken()
    vit: _SettingsViT = _SettingsViT()
    image: _SettingsImage = _SettingsImage()
    paths: _SettingsPaths = _SettingsPaths()

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        env_nested_delimiter = '__'

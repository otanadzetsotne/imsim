import os
from datetime import timedelta
from PIL import Image


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
MODEL_VIT_TYPES = [
    'B_16',
    'B_32',
    'L_32',
    'B_16_imagenet1k',
    'B_32_imagenet1k',
    'L_16_imagenet1k',
    'L_32_imagenet1k',
]
MODEL_VIT_TYPE = 'B_16'

MODEL_VIT_PATH = os.path.join(
    MODEL_DIR,
    f'{MODEL_VIT_TYPE}_{IMAGE_SIZE}.pickle',
)

MODEL_VIT_ENCODER_NAME = 'ViT_encoder'
MODEL_VIT_ENCODER_PATH = os.path.join(MODEL_DIR, f'{MODEL_VIT_ENCODER_NAME}.pickle')

# Base configs

REQUIRED_DIRECTORIES = [
    MODEL_DIR,
]

# Data Types configs

IMAGE_ERR_CODE_OK = 200

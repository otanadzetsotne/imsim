import os

APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Models configs

MODEL_DIR = os.path.join(APP_DIR, 'models')
MODEL_NAME_VIT = 'vit'
MODEL_INPUT = 480
MODEL_INPUT_TINY = 256

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
    f'{MODEL_VIT_TYPE}_{MODEL_INPUT}.pickle',
)
MODEL_VIT_TINY_PATH = os.path.join(
    MODEL_DIR,
    f'{MODEL_VIT_TYPE}_{MODEL_INPUT_TINY}.pickle',
)

MODEL_VIT_ENCODER_NAME = 'ViT_encoder'
MODEL_VIT_ENCODER_PATH = os.path.join(MODEL_DIR, f'{MODEL_VIT_ENCODER_NAME}.pickle')

# Base configs

REQUIRED_DIRECTORIES = [
    MODEL_DIR,
]

# Downloader configs

IMAGE_PIL_FORMAT = 'RGB'
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

# Data Types configs
IMAGE_ERR_CODE_OK = 200


if __name__ == '__main__':
    def directory_exists(path: str):
        """
        Check if directory exists
        :param path: path to directory
        :return: bool
        """

        if os.path.isfile(path):
            raise FileExistsError(path)

        return os.path.exists(path)

    # Create required directories
    directories_checked = []
    for directory in REQUIRED_DIRECTORIES:
        directory = os.path.normpath(directory)

        if directory_exists(directory):
            continue

        # Get all paths
        directories = directory.split(os.sep)
        # Root always exists
        directory_current = directories.pop(0)
        # Check each directory from root
        for directory_to_check in directories:
            directory_current = f'{directory_current}{os.sep}{directory_to_check}'

            if directory_current in directories_checked:
                continue

            if not directory_exists(directory_current):
                os.mkdir(directory_current)

            directories_checked.append(directory_current)

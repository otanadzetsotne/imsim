import os
from typing import Optional


def check_directories(dirs: dict):
    """ Creating directories if they are not exist """

    for dir_ in dirs.values():
        if dir_ is dict:
            check_directories(dir_)

        if not os.path.exists(dir_) and not os.path.isfile(dir_):
            os.mkdir(dir_)


class Configs:
    __configs = {
        # Project Directories
        'directories': {
            'models': os.path.abspath('../models'),
            # Test Directories
            'test': os.path.abspath('../tests'),
            'test_images': os.path.abspath('../tests/data/images.txt'),
            'test_dataset': os.path.abspath('../tests/data/dataset'),
            'test_predictions': os.path.abspath('../tests/data/predictions'),
        },
        # Project Models
        'models': {
            'vit': {
                'name': 'B_32_imagenet1k',
                'image_size': 160,
                'pretrained': True,
            },
            'encoder': {
                'name': 'base',
            },
            'yolov5': {
                'name': 'yolov5s',
            },
        },
    }

    @classmethod
    def check_directories(cls):
        """ Creating directories if they are not exist """

        for dir_ in cls.get('directories'):
            if not os.path.exists(dir_) or not os.path.isdir(dir_):
                os.mkdir(dir_)

    @classmethod
    def get(cls, config: str):
        return cls.__get(cls.__configs, config.split('.'))

    @classmethod
    def __get(cls, configs: Optional[dict], search: list):
        if len(search) == 0 or configs is None:
            return configs

        configs = configs.get(search.pop(0))

        return cls.__get(configs, search)


check_directories(Configs.get('directories'))

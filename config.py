import os


def check_dir(dirs: list):
    """ Creating directories if they are not exist """

    for dir_ in dirs:
        if not os.path.exists(dir_) or not os.path.isdir(dir_):
            os.mkdir(dir_)


path_model = os.path.abspath('model')

check_dir([
    path_model,
])


VIT_IMAGE_SIZE = 160


class Configs:
    __configs = {
        """ Project Directories """
        'directories': {
            'models': os.path.abspath('../models')
        },

        """ Project Models """
        'models': {
            'vit': {
                'name': 'B_32_imagenet1k',
                'image_size': 160,
                'pretrained': True,
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
    def __get(cls, configs: dict, search: list):
        if len(search) == 0 or configs is None:
            return configs

        return cls.__get(configs.get(search[0]), search.pop(0))

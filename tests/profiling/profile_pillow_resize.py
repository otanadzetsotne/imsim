from time import perf_counter
from copy import deepcopy
from PIL import Image


images_quantity = 1000
mock_image = Image.Image.new('RGB', (1000, 1000))
mock_image_list = [deepcopy(mock_image) for _ in range(images_quantity)]


def profile(func):
    def inner(*args, **kwargs):
        tic = perf_counter()
        result = func(*args, **kwargs)
        tac = perf_counter() - tic
        print(f'Time spent in {func.__name__}: {tac} seconds')

        return result

    return inner


def profile_resize(images, resize_filter):
    return [image.resize((256, 256), resize_filter) for image in images]


@profile
def profile_lanczos(images):
    return profile_resize(images, Image.LANCZOS)


@profile
def profile_bicubic(images):
    return profile_resize(images, Image.BICUBIC)


@profile
def profile_bilinear(images):
    return profile_resize(images, Image.BILINEAR)


if __name__ == '__main__':
    profile_lanczos(deepcopy(mock_image))
    profile_bicubic(deepcopy(mock_image))
    profile_bilinear(deepcopy(mock_image))

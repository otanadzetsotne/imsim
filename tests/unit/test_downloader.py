from copy import deepcopy
from src.dtypes import ImageIn, ImageInner, ImagesInner, ImagePIL, ModelInput
from src.utils.downloader import Downloader


images_quantity = 5
mock_url = 'https://ae01.alicdn.com/kf/HTB1OwMBHNGYBuNjy0Fnq6x5lpXab/100-DIY-5D.jpg'
mock_image_in = ImageIn(url=mock_url)
mock_images_in_many = [deepcopy(mock_image_in) for _ in range(images_quantity)]
mock_images_in_one = [deepcopy(mock_image_in)]

mock_model_input_s = ModelInput(192)
mock_model_input_m = ModelInput(480)


def _image_data_test(image):
    # Images downloaded correct
    assert image.pil is not None
    assert isinstance(image.pil, ImagePIL)

    # Error code is OK
    assert image.err.code == 200

    # We have not prediction after downloading
    assert image.prediction is None


def _images_data_test(images):
    for image in images:
        _image_data_test(image)


def test_map_image_data_with_many():
    images = Downloader.map(
        mock_model_input_s,
        mock_images_in_many,
    )

    _images_data_test(images)


def test_map_image_data_with_one():
    images = Downloader.map(
        mock_model_input_s,
        mock_images_in_one,
    )

    _images_data_test(images)


def test_one_image_data():
    image = Downloader.one(
        mock_model_input_s,
        mock_image_in,
    )

    _image_data_test(image)


def test_image_size_s():
    image = Downloader.one(
        mock_model_input_s,
        mock_image_in,
    )

    assert (mock_model_input_s.value, mock_model_input_s.value) == image.pil.size


def test_image_size_m():
    image = Downloader.one(
        mock_model_input_m,
        mock_image_in,
    )

    assert (mock_model_input_m.value, mock_model_input_m.value) == image.pil.size

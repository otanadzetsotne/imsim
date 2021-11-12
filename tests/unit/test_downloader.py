from copy import deepcopy
from src.dtypes import ImageIn, ImageInner, ImagesInner, ImagePIL
from src.utils.downloader import Downloader
from config import IMAGE_SIZE


images_quantity = 5
mock_url = 'https://ae01.alicdn.com/kf/HTB1OwMBHNGYBuNjy0Fnq6x5lpXab/100-DIY-5D.jpg'
mock_image_in = ImageIn(url=mock_url)
mock_images_in_many = [deepcopy(mock_image_in) for _ in range(images_quantity)]
mock_images_in_one = [deepcopy(mock_image_in)]


class TestDownloader:
    @staticmethod
    def _test_data_image(image):
        # Images downloaded correct
        assert image.pil is not None
        assert isinstance(image.pil, ImagePIL)

        # Error code is OK
        assert image.err.code == 200

        # We have not prediction after downloading
        assert image.prediction is None

    def _test_data_images(self, images):
        for image in images:
            self._test_data_image(image)

    def test_map_image_data_with_many(self):
        images = Downloader.map(mock_images_in_many)
        self._test_data_images(images)

    def test_map_image_data_with_one(self):
        images = Downloader.map(mock_images_in_one)
        self._test_data_images(images)

    def test_one_image_data(self):
        image = Downloader.one(mock_image_in)
        self._test_data_image(image)

    def test_image_size(self):
        image = Downloader.one(mock_image_in)
        assert (IMAGE_SIZE, IMAGE_SIZE) == image.pil.size

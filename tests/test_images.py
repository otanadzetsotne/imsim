from src.facades.proxies import ProxyImages
from tests import mock


class TestImagesHelper:
    def test_filter_correct(self):
        assert ProxyImages.filter_correct(mock.images_mix) == mock.images_correct
        assert ProxyImages.filter_correct(mock.images_correct) == mock.images_correct
        assert ProxyImages.filter_correct(mock.images_err) == []
        assert ProxyImages.filter_correct([]) == []

    def test_has_correct(self):
        assert ProxyImages.has_correct(mock.images_mix) is True
        assert ProxyImages.has_correct(mock.images_correct) is True
        assert ProxyImages.has_correct(mock.images_err) is False
        assert ProxyImages.has_correct([]) is False

from abc import ABCMeta

import torch
from pytorch_pretrained_vit import ViT

from src.utils.collector import Collector, Identity
from src.dtypes import Model
from config import MODEL_NAME_VIT, IMAGE_SIZE


mock_vector = torch.randn(IMAGE_SIZE)
mock_vector_4d = torch.randn((2, 3, 4, IMAGE_SIZE))

mock_model_type_vit = Model('vit')


# Test Identity layer


class TestIdentity:
    mock_model_identity = Identity()

    def _test_out(self, vector):
        identity_out = self.mock_model_identity.forward(vector)
        compare_each = torch.eq(vector, identity_out)
        compare_all = torch.all(compare_each)

        return compare_all

    def test_out_1d(self):
        assert self._test_out(mock_vector)

    def test_out_4d(self):
        assert self._test_out(mock_vector_4d)


# Test ViT models


class TestCollectorVit(metaclass=ABCMeta):
    mock_model = Collector.collect(mock_model_type_vit)
    mock_model_vit = dict(mock_model.named_children())[MODEL_NAME_VIT]

    def test_model_collected(self):
        assert self.mock_model is not None
        assert self.mock_model_vit is not None

    def test_model_structure(self):
        assert isinstance(self.mock_model, torch.nn.Module)
        assert isinstance(self.mock_model_vit, ViT)
        assert isinstance(self.mock_model_vit.fc, Identity)

    def test_image_size(self):
        assert self.mock_model_vit.image_size == IMAGE_SIZE

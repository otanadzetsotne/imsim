from abc import ABCMeta

import torch
from pytorch_pretrained_vit import ViT

from src.utils.collector import Collector, Identity
from src.dtypes import Model
from config import MODEL_NAME_VIT, MODEL_INPUT, MODEL_INPUT_TINY


mock_vector = torch.randn(256)
mock_vector_4d = torch.randn((2, 3, 4, 256))
mock_identity = Identity()

mock_model_type = Model.vit
mock_model_type_tiny = Model.vit_tiny

mock_model = Collector.collect(mock_model_type)
mock_model_vit = dict(mock_model.named_children())[MODEL_NAME_VIT]

mock_model_tiny = Collector.collect(mock_model_type_tiny)
mock_model_vit_tiny = dict(mock_model_tiny.named_children())[MODEL_NAME_VIT]


# Test Identity layer


class TestIdentity:
    @staticmethod
    def _test_out(vector):
        identity_out = mock_identity.forward(vector)
        compare_each = torch.eq(vector, identity_out)
        compare_all = torch.all(compare_each)

        return compare_all

    def test_out_1d(self):
        assert self._test_out(mock_vector)

    def test_out_4d(self):
        assert self._test_out(mock_vector_4d)


# Test ViT models


class _TestCollectorVitBase(metaclass=ABCMeta):
    mock_model = None
    mock_model_vit = None

    model_input = None

    def test_model_collected(self):
        assert self.mock_model is not None
        assert self.mock_model_vit is not None

    def test_model_type(self):
        assert isinstance(self.mock_model, torch.nn.Module)
        assert isinstance(self.mock_model_vit, ViT)
        assert isinstance(self.mock_model_vit.fc, Identity)

    def test_image_size(self):
        assert self.mock_model_vit.image_size == self.model_input


class TestCollectorVit(_TestCollectorVitBase):
    mock_model = mock_model
    mock_model_vit = mock_model_vit

    model_input = MODEL_INPUT


class TestCollectorVitTiny(_TestCollectorVitBase):
    mock_model = mock_model_tiny
    mock_model_vit = mock_model_vit_tiny

    model_input = MODEL_INPUT_TINY

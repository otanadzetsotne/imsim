from pytorch_pretrained_vit import ViT
from pytorch_pretrained_vit.model import ViT as modelViT

from src.models.abstract import ModelLoader
from config import IMAGE_SIZE
from config import MODEL_VIT_TYPE
from config import MODEL_VIT_PATH


class ModelLoaderViT(ModelLoader):
    """
    For transporting a neural network to a working object of Model class
    """

    _model = None
    _type = MODEL_VIT_TYPE
    _path = MODEL_VIT_PATH
    _image_size = IMAGE_SIZE

    @classmethod
    def _make(cls) -> modelViT:
        """
        Download model from ViT library to local file storage
        :return: ViT model
        """

        return ViT(
            name=cls._type,
            image_size=cls._image_size,
            pretrained=True,
        )

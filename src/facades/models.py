from src.models.vit import ModelLoaderViT
from src.models.vit_encoder import ModelLoaderViTEncoder


class Models:
    """
    Models access facade
    """

    vit = ModelLoaderViT
    vit_encoder = ModelLoaderViTEncoder

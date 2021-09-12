from torch import nn
from config import (
    MODEL_VIT_ENCODER_PATH,
)


class ViTAutoEncoder(nn.Module):
    def __init__(self):
        super(ViTAutoEncoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential()
        self.encoder.add_module('layer_640', nn.Linear(768, 640))
        self.encoder.add_module('layer_640_activation', nn.PReLU())
        self.encoder.add_module('layer_512', nn.Linear(640, 512))
        self.encoder.add_module('layer_512_activation', nn.PReLU())
        self.encoder.add_module('layer_384', nn.Linear(512, 384))
        self.encoder.add_module('layer_384_activation', nn.PReLU())
        self.encoder.add_module('layer_256', nn.Linear(384, 256))
        self.encoder.add_module('layer_256_activation', nn.PReLU())
        self.encoder.add_module('layer_128', nn.Linear(256, 128))
        self.encoder.add_module('layer_128_activation', nn.PReLU())
        self.encoder.add_module('layer_64', nn.Linear(128, 64))
        self.encoder.add_module('layer_64_activation', nn.Tanh())

        # Decoder
        self.decoder = nn.Sequential()
        self.decoder.add_module('layer_128', nn.Linear(64, 128))
        self.decoder.add_module('layer_128_activation', nn.PReLU())
        self.decoder.add_module('layer_256', nn.Linear(128, 256))
        self.decoder.add_module('layer_256_activation', nn.PReLU())
        self.decoder.add_module('layer_384', nn.Linear(256, 384))
        self.decoder.add_module('layer_384_activation', nn.PReLU())
        self.decoder.add_module('layer_512', nn.Linear(384, 512))
        self.decoder.add_module('layer_512_activation', nn.PReLU())
        self.decoder.add_module('layer_640', nn.Linear(512, 640))
        self.decoder.add_module('layer_640_activation', nn.PReLU())
        self.decoder.add_module('layer_768', nn.Linear(640, 768))
        self.decoder.add_module('layer_768_activation', nn.Tanh())

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class ModelLoaderViTEncoder:
    __path = MODEL_VIT_ENCODER_PATH
    __model = None

    @classmethod
    def get(cls) -> nn.Module:
        if not cls.__exists():
            cls.__model = cls.__make()

        if cls.__model is None:
            cls.__model = cls.__load()

        cls.__model.eval()

        return cls.__model

    @classmethod
    def __make(cls):
        pass

    @classmethod
    def __exists(cls):
        return True

    @classmethod
    def __load(cls):
        return True

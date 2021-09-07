import torch
import fire
from pytorch_pretrained_vit import ViT
from config import (
    MODEL_VIT_NAME,
    MODEL_INPUT,
)


model_path = 'model_vit_save.pickle'


def model_vit_save():
    model = ViT(
        name=MODEL_VIT_NAME,
        image_size=MODEL_INPUT,
    )
    torch.save(model, model_path)


def model_vit_load():
    model = torch.load(model_path)
    print(model)


def run_internal():
    model_vit_save()
    model_vit_load()


if __name__ == '__main__':
    fire.Fire()

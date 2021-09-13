import os
from scipy.spatial import distance
from PIL import Image
from pathlib import Path
import torch
from torchvision import transforms
from src.dtypes import Model
from src.facades.mediators import MediatorCollector
from config import MODEL_INPUT


class _Identity(torch.nn.Module):
    """
    Linear layer for torch model
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        return x


def transform(
        pil,
):
    return transforms.Compose([
        transforms.Resize((MODEL_INPUT, MODEL_INPUT)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])(pil).unsqueeze(0)


def similarities(
        path_similarities: str,
        images: list,
        pillows: list,
        model,
):
    if not os.path.exists(path_similarities):
        os.mkdir(path_similarities)

    path_euclid = os.path.join(path_similarities, 'euclid')
    if not os.path.exists(path_euclid):
        os.mkdir(path_euclid)

    path_cosine = os.path.join(path_similarities, 'cosine')
    if not os.path.exists(path_cosine):
        os.mkdir(path_cosine)

    pillows = pillows[:100]
    prediction_x = model(transform(pillows[0])).detach().numpy()

    for k, pillow in enumerate(pillows):
        prediction = model(transform(pillow)).detach().numpy()

        euclid = str(distance.euclidean(prediction, prediction_x))
        cosine = str(distance.cosine(prediction, prediction_x))

        pillow.save(f'{path_euclid}/{euclid}.jpeg', 'JPEG')
        pillow.save(f'{path_cosine}/{cosine}.jpeg', 'JPEG')


def run(
        path_similarities: str,
        path_images: str,
):
    path_similarities = os.path.join(path_similarities, 'similarities')

    if not os.path.exists(path_similarities):
        os.mkdir(path_similarities)

    model = Model('vit')
    model = MediatorCollector.collect(model)

    i = 0
    images = []
    for image in Path(path_images).glob('*'):
        i += 1
        print('New iteration')

        if i <= 100_000:
            continue

        images.append(str(image))
        print('Image got')

        if i >= 101_000:
            break

    pillows = [Image.open(image).convert('RGB') for image in images]

    model_loc = model.B_16
    similarities(
        os.path.join(path_similarities, 'base'),
        images,
        pillows,
        model_loc,
    )

    similarities(
        os.path.join(path_similarities, 'encoded'),
        images,
        pillows,
        model,
    )


if __name__ == '__main__':
    run(
        os.path.abspath(''),
        'F:\\data\\Datasets\\train2017_max640px',
    )

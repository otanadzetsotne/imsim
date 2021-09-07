import torch
from torchvision import transforms

from src.dtypes import ImagesInner
from config import MODEL_INPUT


class Predictor:
    __transformer = transforms.Compose([
        transforms.Resize((MODEL_INPUT, MODEL_INPUT)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])

    @classmethod
    def predict(
            cls,
            model: torch.nn.Module,
            images: ImagesInner,
    ) -> ImagesInner:
        """
        Predict images
        :param model: Torch model
        :param images: images to predict
        :return: predicted images
        """

        # Pop invalid images
        images_err = [images.pop(images.index(image)) for image in images if image.err is not None]

        # Get images tensor
        tensor = cls.__get_tensor(images)

        # Predict tensors
        predictions = model(tensor)

        # Update images
        for k, prediction in enumerate(predictions):
            images[k].prediction = prediction

        # Concat images lists
        images = images + images_err

        return images

    @classmethod
    def __get_tensor(
            cls,
            images: ImagesInner,
    ) -> torch.Tensor:
        """
        Make tensor from images
        :param images: list of images
        :return: tensor
        """

        # Get images
        pillows = [image.pil for image in images]

        # Tensors from images
        tensors = [cls.__transformer(pillow).unsqueeze(0) for pillow in pillows]

        # Concatenate tensors
        tensor = torch.cat(tensors, 0)

        return tensor

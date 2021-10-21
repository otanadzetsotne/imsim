import torch
from torchvision import transforms

from src.dtypes import ImagesInner
from config import MODEL_INPUT


class Predictor:
    transformer = transforms.Compose([
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

        # Get images tensor
        tensor = cls.__get_tensor(images)

        # Throw tensor to GPU RAM
        if torch.cuda.is_available():
            tensor = tensor.cuda()

        # Predict tensors
        with torch.no_grad():
            predictions = model(tensor)

        # Throw predictions to cpu
        if torch.cuda.is_available():
            predictions = predictions.cpu()

        # Update images
        for k, prediction in enumerate(predictions):
            images[k].prediction = list(prediction)

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
        tensors = [cls.transformer(pillow).unsqueeze(0) for pillow in pillows]

        # Concatenate tensors
        tensor = torch.cat(tensors, 0)

        return tensor

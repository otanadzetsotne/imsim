import fire
from pytorch_pretrained_vit import ViT

from config import MODEL_VIT_NAMES


def model_vit_sizes():
    """
    Print models sizes
    :return:
    """
    for model_name in MODEL_VIT_NAMES:
        model = ViT(
            name=model_name,
            image_size=640,
            pretrained=True,
        )

        print(f'Model: {model_name}')
        print(f'Model parameters: {sum([param.nelement() for param in model.parameters()])}')
        print()


if __name__ == '__main__':
    fire.Fire()

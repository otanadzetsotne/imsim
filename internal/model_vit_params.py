import fire
from pytorch_pretrained_vit import ViT


def model_vit_sizes(
        name: str,
        image_size: int,
        pretrained: bool,
):
    """
    Print models parameters
    :param name: one of ViT model names
    :param image_size: input image size
    :param pretrained: flag for pretrained model
    :return:
    """
    model = ViT(
        name=name,
        image_size=image_size,
        pretrained=pretrained,
    )

    print(f'ViT model {name} with input size {image_size}')
    print(f'Model parameters: {sum([param.nelement() for param in model.parameters()])}')


if __name__ == '__main__':
    fire.Fire()

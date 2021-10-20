from pytorch_pretrained_vit import ViT


MODEL_VIT_TYPE = 'B_16'
MODEL_INPUT = 480


def model_vit_sizes(
        image_size: int = MODEL_INPUT,
        name: str = MODEL_VIT_TYPE,
        pretrained: bool = True,
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

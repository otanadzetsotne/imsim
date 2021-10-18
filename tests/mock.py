from copy import deepcopy

import numpy as np
from PIL import Image as ImagePIL
from torch import Tensor
from src.dtypes import ImageInner


image_data_url = 'https://google.com/image.jpg'
image_data_pil = ImagePIL.new('RGB', (1, 1))
image_data_err = Exception('mock')
image_data_prediction = Tensor(np.ones(768))


image_err = ImageInner(
    url=image_data_url,
    pil=None,
    prediction=None,
    err=image_data_err,
)
image_correct = ImageInner(
    url=image_data_url,
    pil=image_data_pil,
    prediction=None,
    err=None,
)
image_predicted = ImageInner(
    url=image_data_url,
    pil=image_data_pil,
    prediction=image_data_prediction,
    err=None,
)

images_correct = [deepcopy(image_correct) for _ in range(3)]
images_err = [deepcopy(image_err) for _ in range(3)]
images_mix = deepcopy(images_correct) + deepcopy(images_err)


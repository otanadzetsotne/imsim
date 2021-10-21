from copy import deepcopy

import numpy as np
from PIL import Image as ImagePIL
from torch import Tensor
from src.dtypes import ImageInner
from src.dtypes import ImageError
from config import IMAGE_ERR_CODE_OK

image_data_url_bad = 'https://vk.com'
image_data_url_ok = 'https://cdn.vox-cdn.com/thumbor/6ofCV2k-SeKBJytacICi1mrzvL8=/0x66:1600x1133/1200x800/filters:focal(0x66:1600x1133)/cdn.vox-cdn.com/uploads/chorus_image/image/37575328/hello_maybe_not_kitty.0.0.jpeg'

image_data_url = 'https://google.com/image.jpg'
image_data_pil = ImagePIL.new('RGB', (1, 1))
image_data_err_ok = ImageError(code=IMAGE_ERR_CODE_OK)
image_data_err_500 = ImageError(code=500)
image_data_prediction = Tensor(np.ones(768))


image_err_500 = ImageInner(
    url=image_data_url,
    err=image_data_err_500,
    pil=None,
    prediction=None,
)
image_correct = ImageInner(
    url=image_data_url,
    err=image_data_err_ok,
    pil=image_data_pil,
    prediction=None,
)
image_predicted = ImageInner(
    url=image_data_url,
    err=image_data_err_ok,
    pil=image_data_pil,
    prediction=image_data_prediction,
)

images_correct = [deepcopy(image_correct) for _ in range(3)]
images_err = [deepcopy(image_err_500) for _ in range(3)]
images_mix = deepcopy(images_correct) + deepcopy(images_err)


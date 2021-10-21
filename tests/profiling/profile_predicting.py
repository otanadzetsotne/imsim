from copy import deepcopy

import os
import sys

script_path = os.path.realpath(os.path.dirname(__name__))
os.chdir(script_path)
sys.path.append("../..")

from src.dtypes import PredictionInMulti, ImageIn, Model
from src.bl import BusinessLogic
from tests.mock import image_data_url_ok


images_quantity = 25

mock_model_type = Model.vit
mock_image = ImageIn(url=image_data_url_ok)
mock_images = [deepcopy(mock_image) for _ in range(images_quantity)]

mock_request = PredictionInMulti(
    model=mock_model_type,
    images=mock_images,
)


if __name__ == '__main__':
    BusinessLogic.request_predict(mock_request)

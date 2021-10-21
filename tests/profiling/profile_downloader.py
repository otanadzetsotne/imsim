import os
import sys
from copy import deepcopy

script_path = os.path.realpath(os.path.dirname(__name__))
os.chdir(script_path)
sys.path.append("../..")

from src.utils.downloader import Downloader
from src.dtypes import ImageIn, ModelInput

images_quantity = 100
mock_url = 'https://ae01.alicdn.com/kf/HTB1OwMBHNGYBuNjy0Fnq6x5lpXab/100-DIY-5D.jpg'
mock_image = ImageIn(url=mock_url)
mock_images = [deepcopy(mock_image) for _ in range(images_quantity)]

mock_model_input_s = ModelInput(192)
mock_model_input_m = ModelInput(480)


if __name__ == '__main__':
    Downloader.map(mock_model_input_s, mock_images)

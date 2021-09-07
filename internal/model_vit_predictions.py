import pickle
import fire
import os
from itertools import repeat
import PIL
from multiprocessing import Pool
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from src.vit import ModelViT
from config import IMAGE_PIL_FORMAT


class Predictor:
    model = ModelViT()

    def __init__(
            self,
            path_images: str,
            path_predictions: str,
    ):
        self.path_images = path_images
        self.path_predictions = path_predictions

    def predict_one(
            self,
            image_name: str,
    ):
        with Image.open(f'{self.path_images}/{image_name}') as image:
            image = image.convert(IMAGE_PIL_FORMAT)
            prediction = self.model.predict(image)

            with open(f'{self.path_predictions}/{image_name}.pickle', 'wb') as f:
                print(f'Predicted {image_name}')
                pickle.dump(prediction, f)

    def predict_map(
            self
    ):
        # images = os.listdir(self.path_images)
        #
        # with open('list.pickle', 'wb') as f:
        #     pickle.dump(images, f)

        with open('list.pickle', 'rb') as f:
            images = pickle.load(f)

        print('Got images list')
        print(f'Images quantity: {len(images)}')

        print('Start mapping')

        with ThreadPoolExecutor() as executor:
            list(executor.map(self.predict_one, images))

        # images = list(images)


if __name__ == '__main__':
    p_images = 'F:\\data\\Datasets\\google'
    p_predictions = 'C:\\Users\\otana\\Разработка\\py\\imsim_predictions'

    Predictor(p_images, p_predictions).predict_map()

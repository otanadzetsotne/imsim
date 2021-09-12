import pickle
import time
import fire
import os
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

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

    def predict(
            self,
            image,
    ):
        return self.model.predict(image)

    def image(
            self,
            image_name: str,
    ):
        return Image.open(f'{self.path_images}/{image_name}').convert(IMAGE_PIL_FORMAT)

    def predict_map(
            self,
            rest: int,
    ):
        images = os.listdir(self.path_images)[:500_000]

        with open('images_list.pickle', 'wb') as f:
            pickle.dump(images, f)

        print(f'Got images list ({len(images)} images)')

        batch_size = 250
        print('Start mapping')
        for batch_i in range(len(images) // batch_size):
            tic = time.perf_counter()

            print()
            print('New iteration')

            batch_names = images[batch_i * batch_size:batch_i * batch_size + batch_size]
            batch_names = [name for name in batch_names if not os.path.exists(f'{self.path_predictions}/{name}.pickle')]

            if len(batch_names) == 0:
                continue

            with ProcessPoolExecutor() as executor:
                batch_images = executor.map(self.image, batch_names)
            print('Images collected')

            batch_predictions = map(self.predict, batch_images)
            print('Images predictions mapping created')

            for i, prediction in enumerate(batch_predictions):
                with open(f'{self.path_predictions}/{batch_names[i]}.pickle', 'wb') as f:
                    pickle.dump(prediction, f)
            print('Predictions saved')

            tac = time.perf_counter() - tic
            print(f'Prediction time: {batch_size // tac}/1s.')

            time.sleep(rest)


def predict(
        path_images: str,
        path_predictions: str,
        rest: int = 0,
):
    Predictor(path_images, path_predictions).predict_map(rest)


if __name__ == '__main__':
    # predict(
    #     '/media/otana/Remote HDD/data/Datasets/ImageNet_max500px',
    #     '/home/otana/development/py/imsim_predictions/'
    # )
    fire.Fire()

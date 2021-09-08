import pickle
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
            self
    ):
        images = os.listdir(self.path_images)

        print('Got images list')
        print(f'Images quantity: {len(images)}')

        batch_size = 250
        print('Start mapping')
        for batch_i in range(len(images) // batch_size):
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
            print('Images predicted')

            for i, prediction in enumerate(batch_predictions):
                with open(f'{self.path_predictions}/{batch_names[i]}.pickle', 'wb') as f:
                    pickle.dump(prediction, f)
                    print('Predictions saved')


def predict(
        path_images: str,
        path_predictions: str,
):
    Predictor(path_images, path_predictions).predict_map()


if __name__ == '__main__':
    fire.Fire()

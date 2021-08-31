import os
import random
import pickle
import shutil
import copy

from scipy.spatial import distance

from app import Application
from src.datahelpers import ImageDataCreator


path_images = os.path.abspath('provider_images')
path_predictions = os.path.abspath('predictions')
# path_images_list_test = os.path.abspath('images_list_test.pickle')
# path_images_list_train = os.path.abspath('images_list_train.pickle')

path_similarities = os.path.abspath('similarities')
path_similarities_base = f'{path_similarities}/base'
path_similarities_encoder = f'{path_similarities}/encoder'


if __name__ == '__main__':
    def create_predictions(application, images, prefix):
        images = [ImageDataCreator.create_by_path(f'{path_images}/{image_name}') for image_name in images]
        images_predicted = application.predict(images)

        predictions = []

        for k, image in enumerate(images_predicted):
            if image.err is not None:
                print(f'Error: {image.err}')
                images_predicted.pop(k)

            predictions.append(image.get_data_mean())

        x_target = list(predictions[0].reshape(-1))

        for k, prediction in enumerate(predictions):
            x = list(prediction.reshape(-1))

            dist = str(float(distance.euclidean(x_target, x)))

            image_path = images_predicted[k].path
            # print(f'{path_similarities}/{prefix}/{dist}.jpg')
            shutil.copy(image_path, f'{path_similarities}/{prefix}/{dist}.jpg')


    # with open('images_list.pickle', 'rb') as f:
    #     images_list = pickle.load(f)[:10]

    images_list = os.listdir(path_images)
    images_list = random.sample(images_list, 100)

    create_predictions(Application(False, False, 160, 'yolov5s', 310, 'B_16'), copy.copy(images_list), 'base')
    create_predictions(Application(False, True, 160, 'yolov5s', 310, 'B_16'), copy.copy(images_list), 'encoder')

import os
import pickle
import multiprocessing
import shutil
import numpy as np
from scipy.spatial.distance import cdist

import test_config as c
from model import Model


def predict(file):
    path_file = f'{c.path_dataset}/{file}'
    path_x = f'{c.path_predictions_std_pickles}/{file}.pickle'

    if os.path.exists(path_x):
        return

    try:
        x = Model()(path_file, 'path')
        x = np.array(x)
        with open(path_x, 'wb') as f:
            pickle.dump(x, f)
    finally:
        return


def predict_all(files):
    pool = multiprocessing.Pool(3)
    pool.map(predict, files)
    pool.close()


def predict_all_similarities(files):
    file_t = files[3]
    path_x_t = f'{c.path_predictions_std_pickles}/{file_t}.pickle'

    with open(path_x_t, 'rb') as f:
        x_t = pickle.load(f)

    for file in files:
        try:
            path_file = f'{c.path_dataset}/{file}'
            path_x = f'{c.path_predictions_std_pickles}/{file}.pickle'

            with open(path_x, 'rb') as f:
                x = pickle.load(f)

            euclidean = str(float(cdist(x_t, x, 'euclidean')))
            path_similarity = f'{c.path_predictions_std_similarities}/{euclidean}.jpg'

            shutil.copy(path_file, path_similarity)
        finally:
            continue


if __name__ == '__main__':
    images = os.listdir(c.path_dataset)[:10]
    predict_all(images)
    predict_all_similarities(images)

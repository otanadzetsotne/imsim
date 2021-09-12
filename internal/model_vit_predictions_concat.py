import os
import pickle

import fire
import numpy as np
from collections import deque


def concat(
        path_predictions: str,
        path_predictions_concatenated: str,
):
    """
    :param path_predictions: path to pickled predictions vectors
    :param path_predictions_concatenated: path to concatenated predictions matrix file
    """

    predictions = os.listdir(path_predictions)
    predictions_quantity = len(predictions)
    predictions = map(lambda x: f'{path_predictions}/{x}', predictions)
    print('Got predictions')

    predictions_concat = deque()
    for prediction in predictions:
        with open(prediction, 'rb') as f:
            predictions_concat.append(pickle.load(f))
    print('Got predictions list')

    predictions_concat = np.array(predictions_concat)
    print('Created predictions np.array')
    predictions_concat = predictions_concat.reshape((predictions_quantity, -1))
    print(f'Reshaped predictions np.array {predictions_concat.shape}')

    with open(path_predictions_concatenated, 'wb') as f:
        pickle.dump(predictions_concat, f)
    print(f'Saved predictions np.array ({path_predictions_concatenated})')


if __name__ == '__main__':
    fire.Fire()

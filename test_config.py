import os
import config as c

path_test = os.path.abspath('test')
path_dataset = f'{path_test}/dataset'
path_predictions = f'{path_test}/predictions'
path_predictions_std = f'{path_predictions}/standard'
path_predictions_std_pickles = f'{path_predictions_std}/pickles'
path_predictions_std_similarities = f'{path_predictions_std}/similarities'

c.check_dir([
    path_test,
    path_dataset,
    path_predictions,
    path_predictions_std,
    path_predictions_std_pickles,
    path_predictions_std_similarities,
])

import os


def check_dir(dirs: list):
    for dir_ in dirs:
        if not os.path.exists(dir_) or not os.path.isdir(dir_):
            os.mkdir(dir_)


path_model = os.path.abspath('model')

check_dir([
    path_model,
])

import os

APP_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(APP_DIR, 'models')

MODEL_VIT_DIR = os.path.join(MODEL_DIR, 'vit')
MODEL_VIT_INPUT = 720
MODEL_VIT_NAME = 'B_16'


REQUIRED_DIRECTORIES = [
    MODEL_DIR,
    MODEL_VIT_DIR,
]


if __name__ == '__main__':
    def directory_exists(path: str):
        """
        Check if directory exists
        :param path: path to directory
        :return: bool
        """

        if os.path.isfile(directory_current):
            raise FileExistsError(directory_current)

        return os.path.exists(path)

    # Create required directories
    directories_checked = []
    for directory in REQUIRED_DIRECTORIES:
        directory = os.path.normpath(directory)

        if directory_exists(directory):
            continue

        # Get all paths
        directories = directory.split(os.sep)
        # Root always exists
        directory_current = directories.pop(0)
        # Check each directory from root
        for directory_to_check in directories:
            directory_current = f'{directory_current}{os.sep}{directory_to_check}'

            if directory_current in directories_checked:
                continue

            if not directory_exists(directory_current):
                os.mkdir(directory_current)

            directories_checked.append(directory_current)

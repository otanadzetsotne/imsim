import os
from config import REQUIRED_DIRECTORIES


class TestConfig:
    def test_directories(self):
        for directory in REQUIRED_DIRECTORIES:
            assert os.path.exists(directory)

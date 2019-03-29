import sys
import pytest
import os

PYTHON_VERSION = 3
TITANIC_PATH = os.path.join("data", "raw")
DATA_FILES = ['train.csv', 'test.csv']


def test_python_version():
    """Test for python 3 major version"""
    assert PYTHON_VERSION == sys.version_info.major


def test_raw_data():
    for file in DATA_FILES:
        filename = os.path.join(TITANIC_PATH, file)
        assert os.path.exists(filename)

import sys
import pytest

PYTHON_VERSION = 3


def test_version():
    assert PYTHON_VERSION == sys.version_info.major

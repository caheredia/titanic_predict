import sys
import pytest

PYTHON_VERSION = 3


def test_python_version():
    """Test for python 3 major version"""
    assert PYTHON_VERSION == sys.version_info.major

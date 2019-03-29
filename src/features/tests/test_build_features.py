import pytest
import pandas as pd
from src.features.build_features import load_data


def test_loader():
    assert type(load_data("train.csv")) == type(pd.DataFrame())
    assert type(load_data("test.csv")) == type(pd.DataFrame())

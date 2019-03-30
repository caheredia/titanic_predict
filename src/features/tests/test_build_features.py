import pytest
import pandas as pd
import numpy as np
from src.features.build_features import load_data
from src.features.build_features import num_pipeline

train_data = load_data("train.csv", nrows=5)
test_data = load_data("test.csv", nrows=5)


def test_loader():
    assert type(train_data) == type(pd.DataFrame())
    assert type(test_data) == type(pd.DataFrame())


def test_num_pipeline():
    X = num_pipeline.fit_transform(train_data['Age'].values.reshape(1, -1))
    assert type(X) == type(np.empty(1))

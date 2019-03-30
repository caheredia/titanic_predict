# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# for debugging
logger = logging.getLogger(__name__)
logger.info('Setting up relative path')

TITANIC_PATH = os.path.join("data", "raw")


def load_data(filename, titanic_path=TITANIC_PATH, nrows=None):
    """Loads data into pandas csv.

    Parameters
    ----------
    filename : string
        location of data file 
    titanic_path = string (optional)
        path to data files 
    nrows = int (optional)
        number of rows to load 

    Returns
    -------
    df : pandas.core.frame.DataFrame
        data

     """
    csv_path = os.path.join(titanic_path, filename)
    logger.debug('Loading csv into pandas dataframe')
    return pd.read_csv(csv_path, nrows=nrows)


def add_rel_features(df, column_names, relatives=True):
    '''Adds extra features to data sets.

    If boolean is true adds relatives on board.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        dataframe to modify
    column_names : list
        columns containing relative data

    Returns
    -------
    df : pandas.core.frame.DataFrame
        transformed df .
    '''
    logger.debug('Adding RelativesOnboard column')
    if relatives:
        df['RelativesOnboard'] = df[column_names].sum(axis=1)
        return df


def add_travel_alone(df):
    """Adds traveling alone column to data."""
    logger.debug('Adding traveling_alone column')
    df['traveling_alone'] = np.where(df['RelativesOnboard'] == 0, 1, 0)
    return df


def add_AgeBucket_feature(df, column_name='Age', bin_size=15, add=True):
    '''Adds extra feature to data.

    If boolean is true adds age bucket.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        dataframe to modify
    column_name : string (optional)
        columns containing relative data
    bin_size = int (optional)
        number of categories 
    add: boolean (optional)
        trigger for feature 

    Returns
    -------
    df : pandas.core.frame.DataFrame
        transformed df .
    '''
    logger.debug('Adding Age column')
    if add:
        df['AgeBucket'] = df[column_name] // bin_size * bin_size
        return df


# Inspired from stackoverflow.com/questions/25239958
# returns the most frequent item for each selected column
# fills any nulls with most frequent item
class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],
                                        index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)


# numerical pipeline
num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])

# categorical pipeline
cat_pipeline = Pipeline([
    ("imputer", MostFrequentImputer()),
    ("cat_encoder", OneHotEncoder(sparse=False)),
])

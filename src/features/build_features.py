# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import logging

# for debugging
logger = logging.getLogger(__name__)
logger.info('Setting up relative path')

TITANIC_PATH = os.path.join("data", "raw")


def load_data(filename, titanic_path=TITANIC_PATH):
    """Loads data into pandas csv.

     """
    csv_path = os.path.join(titanic_path, filename)
    logger.debug('Loading csv into pandas dataframe')
    return pd.read_csv(csv_path)


def add_num_features(df, column_names, relatives=True):
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
    df : numpy.Array
        transformed df .
    '''
    logger.debug('Adding RelativesOnboard column')
    if relatives:
        df['RelativeOnboard'] = df[column_names].sum(axis=1)
        return df

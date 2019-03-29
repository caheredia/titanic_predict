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
        df['RelativeOnboard'] = df[column_names].sum(axis=1)
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

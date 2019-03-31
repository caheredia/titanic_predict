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





def add_travel_alone(df):
    """Adds traveling alone column to data."""
    logger.debug('Adding traveling_alone column')
    df['traveling_alone'] = np.where(df['RelativesOnboard'] == 0, 0, 1)
    return df


def add_bucket(df_column, bins=4):
    """Adds binnned bucket column to data.

    Parameters
    ----------
    df_column : pandas.core.frame.DataFrame
        dataframe column containing data to bin
    bins = int (optional)
        number of categories 

    Returns
    -------
    data : pandas.core.frame.DataFrame
        transformed df with labels
    """
    logger.debug('Generating new bucket column')
    data = pd.qcut(df_column, bins, labels=False)
    return data

def set_title(name):
    """Returns the Title in the name string."""
    titles = {
        "Capt.": "Prestige",
        "Col.": "Prestige",
        "Major.": "Prestige",
        "Jonkheer.": "Prestige",
        "Don.": "Prestige",
        "Dona.": "Prestige",
        'Countess.':'Prestige',
        "Sir.": "Prestige",
        "Dr.": "Prestige",
        "Rev.": "Prestige",
        "the. Countess": "Prestige",
        "Mme.": "Mrs",
        "Mlle.": "Miss",
        "Ms.": "Mrs",
        "Mrs.": "Mrs",
        "Mr.": "Mr",
        "Miss.": "Miss",
        "Master.": "Prestige",
        "Lady.": "Prestige"
    }
    
    for key in titles:
        if key in name.split():
            return titles[key]
        



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

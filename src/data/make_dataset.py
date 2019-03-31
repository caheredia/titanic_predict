# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import logging
from src.features.build_features import load_data, set_title, add_bucket, add_rel_features, add_travel_alone
from src.features.build_features import num_pipeline, cat_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.externals import joblib


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Making final data set from raw data')

    # Load data
    train_data = load_data("train.csv")
    test_data = load_data("test.csv")

    # add columns
    logger.info('Adding columns to data sets')
    relatives = ['SibSp', 'Parch']

    def add_columns(df):
        df['RelativesOnboard'] = df[relatives].sum(axis=1)
        df['Age_Bucket'] = add_bucket(df['Age'], bins=6)
        df['Fare_Bucket'] = add_bucket(df['Fare'], bins=6)
        df['Title'] = df['Name'].apply(set_title)
        df['Name_length'] = df['Name'].apply(len)
        df['Cabin'] = df['Cabin'].apply(lambda x: 0 if type(x) == float else 1)
        return df
    train_data = add_columns(train_data)
    test_data = add_columns(test_data)

    # Full pipeline
    logger.info('Applying pipeline to data sets')
    cat_attribs = ["Pclass", 'Embarked',  'Age_Bucket',
                   'Fare_Bucket', 'Title', 'Sex', 'traveling_alone', 'Cabin']
    num_attribs = ["RelativesOnboard", 'Fare', 'Age', 'Name_length']

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs),
    ])

    X_train = full_pipeline.fit_transform(train_data)
    X_test = full_pipeline.transform(test_data)

    # Save the transformed data, test data, and pipeline parameters
    logger.info('Saving transformed data sets and pipeline parameters')
    PROCESSED_PATH = os.path.join("data", "processed")
    # pipeline
    joblib.dump(full_pipeline, os.path.join(
        PROCESSED_PATH, 'full_pipeline.pkl'))
    # X_train
    joblib.dump(X_train, os.path.join(PROCESSED_PATH, 'X_train.pkl'))
    # X_test
    joblib.dump(X_test, os.path.join(PROCESSED_PATH, 'X_test.pkl'))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename=os.path.abspath(
        'titanic.log'), level=logging.DEBUG, format=log_fmt, filemode='w')

    main()

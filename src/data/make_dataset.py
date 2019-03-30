# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import logging
from src.features.build_features import load_data
from src.features.build_features import add_rel_features
from src.features.build_features import add_travel_alone
from src.features.build_features import add_AgeBucket_feature
from src.features.build_features import num_pipeline
from src.features.build_features import cat_pipeline
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

    # add columns
    logger.info('Adding columns to training set')
    relatives = ['SibSp', 'Parch']

    def add_columns(df):
        df = add_rel_features(df, relatives)
        df = add_travel_alone(df)
        df = add_AgeBucket_feature(df)
        return df
    train_data = add_columns(train_data)

    # Full pipeline
    logger.info('Applying pipeline to training set')
    cat_attribs = ["Pclass", "Sex", 'Embarked',
                   'traveling_alone',  'AgeBucket']
    num_attribs = ["RelativesOnboard", "Fare"]

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs),
    ])

    X_train = full_pipeline.fit_transform(train_data)

    # Save the transformed data, test data, and pipeline parameters
    logger.info('Saving transformed data, test data, and pipeline parameters')
    PROCESSED_PATH = os.path.join("data", "processed")
    # pipeline
    joblib.dump(full_pipeline, os.path.join(
        PROCESSED_PATH, 'full_pipeline.pkl'))
    # X_train
    joblib.dump(X_train, os.path.join(PROCESSED_PATH, 'X_train.pkl'))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename=os.path.abspath(
        'titanic.log'), level=logging.DEBUG, format=log_fmt, filemode='w')

    main()

# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import logging
from src.features.build_features import load_data


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Making final data set from raw data')

    train_data = load_data("train.csv")
    logger.debug('Loaded csv file to pandas dataframe.')

    print(train_data.head())


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename=os.path.abspath(
        'titanic.log'), level=logging.DEBUG, format=log_fmt, filemode='w')

    main()

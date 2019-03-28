# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import logging


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    TITANIC_PATH = os.path.join("data", "raw")

    def load_data(filename, titanic_path=TITANIC_PATH):
        csv_path = os.path.join(titanic_path, filename)
        return pd.read_csv(csv_path)

    train_data = load_data("train.csv")

    print(train_data.head())


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename=os.path.abspath(
        'data.Log'), level=logging.DEBUG, format=log_fmt, filemode='w')

    main()

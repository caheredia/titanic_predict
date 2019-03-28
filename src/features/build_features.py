# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import logging


logger = logging.getLogger(__name__)
logger.info('Setting up relative path')

TITANIC_PATH = os.path.join("data", "raw")


def load_data(filename, titanic_path=TITANIC_PATH):
    """Loads data into pandas csv.

     """
    csv_path = os.path.join(titanic_path, filename)
    logger.debug('Loading csv into pandas dataframe')
    return pd.read_csv(csv_path)

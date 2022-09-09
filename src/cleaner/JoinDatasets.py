"""
Dataset Combiner
"""

# ------------- Libraries -------------
import logging

from os import walk

import pandas as pd

# Set logging level
logging.basicConfig(level=logging.INFO)

# ------------- Constants -------------

DATA_DIRECTORIES = ['../../datasets/news/clean', '../../datasets/price/clean', '../scraper/datasets/news/clean', '../scraper/datasets/price/clean']

# ------------- Runner -------------

if __name__ == "__main__":
    files = []

    for directory in DATA_DIRECTORIES:
        for (dirpath, dirnames, filenames) in walk(directory):
            filepaths = []
            for filename in filenames:
                filepaths = filepaths + [f'{directory}/{filename}']
            files = files + filepaths
    
    for file in files:
        logging.info(file)


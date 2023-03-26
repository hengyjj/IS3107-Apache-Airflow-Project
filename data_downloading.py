# ----------------------------- Start of Downloading the data  -----------------------------
# Using Yi Jie's Kaggle API Key
import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Set up API credentials
api = KaggleApi()
api.authenticate()

# Downloading the raw data set
api.dataset_download_files('jiashenliu/515k-hotel-reviews-data-in-europe', path='data', unzip=True)
# ---------------------------- End of Downloading the data  -----------------------------


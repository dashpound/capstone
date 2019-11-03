# ===============================================================================
# 05.00.01 | Clean Data Load | Documentation
# ===============================================================================
# Name:               05_clean_data_load
# Author:             Rodd
# Last Edited Date:   10/20/19
# Description:        Loads cleaned product data for modeling/further analysis.
#                     Coming soon - Loads cleaned reviewer data for modeling/further analysis..
#                     Coming soon - Loads cleaned reviews data for modeling/further analysis..
# Notes:
# Warnings:           
#                     
# Outline:
#                     Unpack cleaned product data data.
#                     Verify cleaned product data loaded successfully.
#                     Unpack one-hot encoded product data.
#                     Verify one-hot encoded product data loaded successfully.
#
#
# =============================================================================
# 05.00.02 | Import Modules & Packages
# =============================================================================
# Import modules (other scripts)
from environment_configuration import working_directory
from environment_configuration import data_path
from environment_configuration import products_clean_path, products_onehot_path

# Import packages
import pickle
import pandas as pd
from pathlib import Path


# Create trigger for turning off verify portions of code
verify = 'n'

print('Script: 05.00.02 [Import Packages] completed')


# =============================================================================
# 05.01.01 | Load Cleaned Product Data
# =============================================================================
# Unpack cleaned product data to use for visualization
with open(Path(working_directory + data_path + products_clean_path), 'rb') as pickle_file:
    products_clean = pickle.load(pickle_file)
    products_clean = pd.DataFrame(products_clean)

print('Script: 05.01.01 [Load Cleaned Product Data] completed')


# =============================================================================
# 05.02.01 | Load One-Hot Encoded Product Data
# =============================================================================
# Unpack product data with one-hot-encodings for modeling
with open(Path(working_directory + data_path + products_onehot_path), 'rb') as pickle_file:
    products_df = pickle.load(pickle_file)
    products_df = pd.DataFrame(products_df)

print('Script: 05.02.01 [Load One Hot Encoded Product Data] completed')
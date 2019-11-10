# ===============================================================================
# 02.00.01 | Data Load | Documentation
# ===============================================================================
# Name:               02_data_load
# Author:             Julia
# Last Edited Date:   10/10/19
# Description:        Loads review data and verifies data loading.
#                     Loads product metadata and verifies data loading.
#                     Loads Q/A data and verifies data loading.
# Notes:
# Warnings:           The data checks are not automated right now but 
#                     they are a starting point.
# Outline:
#                     Unpack review data.
#                     Verify review data loaded successfully.
#                     Unpack product metadata.
#                     Verify product metadata loaded successfully.
#                     Unpack Q/A data.
#                     Verify Q/A data loaded successfully.
# =============================================================================
# 02.00.02 | Import Modules & Packages
# =============================================================================
# Import modules (other scripts)
from code.configuration.environment_configuration import working_directory
from code.configuration.environment_configuration import data_path
from code.configuration.environment_configuration import reviews_path, metadata_path, qa_path

# Import packages
import pickle
import pandas as pd
from pathlib import Path
import gc

# Create trigger for turning off verify portions of code
verify = 'n'

print('Script: 02.00.02 [Import Packages] completed')

# =============================================================================
# 02.01.01 | Load Reviews Data
# =============================================================================
# Unpack review data where reviewer has left at least 5 reviews

with open(Path(working_directory+'/data'+reviews_path), 'rb') as pickle_file:
    reviews_df = pickle.load(pickle_file)
    reviews_df = pd.DataFrame(reviews_df)

print('Script: 02.01.01 [Load Reviews Data] completed')

# =============================================================================
# 02.01.02 | Verify Reviews Data Loaded Successfully
# =============================================================================
if verify == 'n':
    print('Script: 02.01.02 [Verify Data Load] skipped')
else:
    print('Review data shape: ', reviews_df.shape)
    print('Review data columns: ', reviews_df.columns)
    print(reviews_df.head())
    print('Script: 02.01.02 [Verify Data Load] completed')

# =============================================================================
# 02.01.03 | Load Product Metadata Data
# =============================================================================
# Performing garbage collect just in case
gc.collect()

# Unpack metadata
with open(Path(working_directory + data_path + metadata_path), 'rb') as pickle_file:
    metadata_df = pickle.load(pickle_file)
    metadata_df = pd.DataFrame(metadata_df)

print('Script: 02.01.03 [Load Product Metadata Data] completed')

# =============================================================================
# 02.01.04 | Verify Reviews Data Loaded Successfully
# =============================================================================

if verify == 'n':
    print('Script: 02.01.04 [Verify Reviews Data] skipped')
else:
    print('Metadata data shape: ', metadata_df.shape)
    print('Metadata data columns: ', metadata_df.columns)
    print(metadata_df.head())
    print('Script: 02.01.04 [Verify Reviews Data] completed')

# =============================================================================
# 02.01.04 | Load QA Data
# =============================================================================
# Performing garbage collect just in case
gc.collect()

# Unpack QA data
# Not sure if QA data will be used just yet
with open(Path(working_directory + data_path + qa_path), 'rb') as pickle_file:
    qa_df = pickle.load(pickle_file)
    qa_df = pd.DataFrame(qa_df)

print('Script: 02.01.05 [Load QA Data] completed')

# =============================================================================
# 02.01.05 | Verify QA Data Loaded Successfully
# =============================================================================
if verify == 'n':
    print('Script: 02.01.05 [Verify QA Data] skipped')
else:
    print('Q/A data shape: ', qa_df.shape)
    print('Q/A data columns: ', qa_df.columns)
    print(qa_df.head())
    print('Script: 02.01.05 [Verify QA Data] completed')

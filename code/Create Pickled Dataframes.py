# ===============================================================================
# 04.00.01 | Load Raw Review Data & Pickle For Future Use | Documentation
# ===============================================================================
# Name:               Load Raw Review Data & Pickle For Future Use
# Author:             Hemant Patel
# Last Edited Date:   10/07/19
# Description:        Load raw Amazon review data, flatten into dataframe, then create pickled output.
#                     
# Notes:                              
# Warnings:
# Outline:            Imports needed packages and objects.
#                     Extracts raw reviews data and creates flattend dataframe.
#                     Creates pickled files which are then and saved to the data folder in github.
#
# =============================================================================
# 04.00.02 | Import Modules & Packages
# =============================================================================

# Import packages
import pandas as pd
import numpy as np
import json
import os
from pandas.io.json import json_normalize
import gzip
import time

# =============================================================================
# 04.00.03 | Import Raw Reviews Data (for users who left at least 5 electronics reviews)
# =============================================================================

# Start clock 
start = time.time()

# Define directory
path = os.path.dirname(os.path.realpath('/Users/Hemant/Desktop/Amazon Review Data/'))

# Read data into a dataframe
def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

reviewsdata_5 = getDF('reviews_Electronics_5.json.gz')

# Stop clock
end = time.time()

# Examine data output
print('Number of Records:', len(reviewsdata_5))
print('Execution Time In Seconds:', round(end - start, 0))

# =============================================================================
# 04.00.04 | Import Raw Metadata
# =============================================================================

# Start clock 
start = time.time()

# Define directory
path = os.path.dirname(os.path.realpath('/Users/Hemant/Desktop/Amazon Review Data/'))

# Read data into a dataframe
def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

metadata = getDF('meta_Electronics.json.gz')

# Stop clock
end = time.time()

# Examine data output
print('Number of Records:', len(metadata))
print('Execution Time In Seconds:', round(end - start, 0))

# =============================================================================
# 04.00.05 | Import Raw Question/Answer Data
# =============================================================================

# Load metadata for users who left electronics reviews

# Start clock 
start = time.time()

# Define directory
path = os.path.dirname(os.path.realpath('/Users/Hemant/Desktop/Amazon Review Data/'))

# Read data into a dataframe
def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

QA_data = getDF('qa_Electronics.json.gz')

# Stop clock
end = time.time()

# Examine data output
print('Number of Records:', len(QA_data))
print('Execution Time In Seconds:', round(end - start, 0))

# =============================================================================
# 04.00.06 | Create pickled outputs for all dataframes
# =============================================================================

reviewsdata_5.to_pickle('./reviewsdata_5.pkl')
metadata.to_pickle('./metadata.pkl')
QA_data.to_pickle('./QA_data.pkl')
print('Step complete!')

# =============================================================================
# 
# =============================================================================
# ===============================================================================
# 03.00.01 | EDA | Documentation
# ===============================================================================
# Name:               eda
# Author:             Kiley
# Last Edited Date:   10/11/19
# Description:        Load in data frames from data load
#                     Perform basic EDA
# Notes:
# Warnings:
#
# Outline:
#
# =============================================================================
# 03.00.02 | Import Modules & Packages
# =============================================================================
# Import packages
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import gc

# Import modules (other scripts)
from data_load import reviews_df

print('Script: 03.00.02 [Import Packages] completed')

# =============================================================================
# 03.01.01 | Reviews: High Level EDA
# =============================================================================
# Return info on reviews_df
reviews_df.info()

# About dataset
s = reviews_df.shape
print('----------------------------------------------------\n')
print('Reviews dataset is', s[0], 'rows by', s[1], 'columns','\n')
print('----------------------------------------------------\n')
print('Script: 03.01.01 [Reviews: High Level EDA ] completed')

# =============================================================================
# 03.01.02 | Reviews: Reviewer Centric EDA
# =============================================================================

# For each unique reviewer count reviews, calculate average overall,
# Find min & max overall ratings
base_pivot = pd.pivot_table(reviews_df, index=['reviewerID'], values=['overall'],
                            aggfunc={'overall': [len, np.min, np.mean, np.median, np.max]})

# Convert Base Pivot to Data Frame
base_pivot.columns = base_pivot.columns.droplevel(0)
base_pivot = base_pivot.reset_index().rename_axis(None, axis=1)
base_pivot = pd.DataFrame(base_pivot)

# Calculate the average review per product in this category
base_avg = base_pivot['mean'].mean()
# Average number of reviews per product
base_count = base_pivot['len'].mean()

print('----------------------------------------------------\n')
print('Average rating issued by user: {:.2f}'.format(base_avg))
print('Average number of reviews per issued by user {:.2f}'.format(base_count))
print('Number of unique Amazon Reviewers: {:.2f}'.format(base_pivot.shape[0]),'\n')
print('----------------------------------------------------\n')

print(base_pivot)

print('Script: 03.01.02 [Reviews: Reviewer Focused EDA ] completed')

# =============================================================================
# 03.01.03 | Reviews: Product Centric EDA
# =============================================================================

# For each unique product count reviews, calculate average overall
review_pivot = pd.pivot_table(reviews_df, index=['asin'], values=['overall'],
                              aggfunc={'overall': [len, min, np.mean, np.median, max]})

# Convert Review Pivot to Data Frame
review_pivot.columns = review_pivot.columns.droplevel(0)
review_pivot = review_pivot.reset_index().rename_axis(None, axis=1)
review_pivot = pd.DataFrame(review_pivot)

# Calculate the average review per product in this category
product_avg = review_pivot['mean'].mean()
# Average number of reviews per product
product_count = review_pivot['len'].mean()

print('----------------------------------------------------\n')
print('Average product rating: {:.2f}'.format(product_avg))
print('Average reviews per product: {:.2f}'.format(product_count))
print('Number of unique products reviewed: {:.2f}'.format(review_pivot.shape[0]),'\n')
print('----------------------------------------------------\n')
print('Script: 03.01.03 [Reviews: Product Focused EDA ] completed')

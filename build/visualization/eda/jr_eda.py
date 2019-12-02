# ===============================================================================
# 04.00.01 | Julia EDA | Documentation
# ===============================================================================
# Name:               jr eda
# Author:             Rodd
# Last Edited Date:   10/16/19
# Description:        Analyze reviewer and product data to inform data prep/modeling.
#                     
# Notes:
# Warnings:
#
# Outline:
#
# =============================================================================
# 04.00.02 | Import Modules & Packages
# =============================================================================
# Import packages
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import gc

# Import modules (other scripts)
from code.dataprep.data_load import reviews_df, metadata_df
from code.configuration.environment_configuration import set_palette
from code.configuration.functions import conv_pivot2df


print('Script: 04.00.02 [Import Packages] completed')

# =============================================================================
# 04.01.01 | Reviews: High Level EDA
# =============================================================================
# Return info on reviews_df
reviews_df.info()

# convert to timestamps
reviews_df['reviewTimeDate'] = pd.to_datetime(reviews_df['reviewTime'],format='%m %d, %Y')
reviews_df['reviewYear'] = reviews_df['reviewTimeDate'].dt.year


# get year of last review for each reviewer
reviewer_pivot = pd.pivot_table(reviews_df, index=['reviewerID'], 
                                values=['reviewYear'],
                                aggfunc='max')

reviewer_date_count = reviewer_pivot.groupby('reviewYear').size().to_frame('reviewers').reset_index()

# get the number of reviews by reviewer
base_pivot = pd.pivot_table(reviews_df, index=['reviewerID'], values=['overall'],
                            aggfunc={'overall': [len, np.min, np.mean, np.median, np.max]})

conv_pivot2df(base_pivot)

number_of_reviews = base_pivot.groupby('len').size().to_frame('reviewers').reset_index()

# what is reviewer distribution taking the more recent reviewers?
reviews_df2 = reviews_df[reviews_df['reviewYear'].isin([2013, 2014])]
a = reviews_df2['reviewerID'].unique()
reviews_df3 = pd.DataFrame(a, columns=["reviewerID"])

# we want to get all the reviews from the reviewers that had a review in 2013/2014
# something gets messed up doing this - len is off
reviews_filtered = pd.merge(reviews_df, reviews_df3, on='reviewerID', how='inner')

base_pivot2 = pd.pivot_table(reviews_filtered, index=['reviewerID'], values=['overall'],
                            aggfunc={'overall': [len, np.min, np.mean, np.median, np.max]})

conv_pivot2df(base_pivot2)

number_of_reviews2 = base_pivot2.groupby('len').size().to_frame('reviewers').reset_index()

# show results
print('----------------------------------------------------\n')
print('Date of last review: \n',reviewer_date_count,'\n')
print('Reviewer distribution: \n',number_of_reviews,'\n')
print('Reviewer distribution for 2013/2014 reviewers: \n',number_of_reviews2,'\n')
print('----------------------------------------------------\n')
print('Reduction in reviewers: ', base_pivot.shape[0] - base_pivot2.shape[0],'\n')
print('Reduction in reviews: ', reviews_df.shape[0] - reviews_filtered.shape[0])


# =============================================================================
# 04.01.02 | Products: High Level EDA
# =============================================================================
# Return info on metadata_df
metadata_df.info()
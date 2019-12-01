# ===============================================================================
# 12.00.01 | Cluster | Documentation
# ===============================================================================
# Name:               12_baseline_ratings_model_clusters
# Author:             Kiley
# Last Edited Date:   11/17/19
# Description:        Top N by cluster
#
# Notes:              TODO: Only set up for cluster 0 run this for every cluster
#
#
# Warnings:           Experiencing memory issues building predictions. Tested code using 1% of data.
#
#
# Outline:            Imports needed packages and set seed for reproducibility.
#
#
# =============================================================================
# 12.00.02 | Import Packages
# =============================================================================
# Import packages
import pandas as pd
import numpy as np
import random
import gc
import pickle
from pathlib import Path
from timeit import default_timer as timer

from build.configuration.environment_configuration import RANDOM_SEED, working_directory

clusters=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

print('Script: 12.00.02 [Import packages] completed')

# =============================================================================
# 12.00.03 | Readin predictions - SVD
# =============================================================================
start = timer()
cluster = clusters[0]

with open((working_directory + '/output' + '/baseline_model/svd_recommendations_cluster'+cluster+'.pkl'), 'rb') as pickle_file:
    rec = pickle.load(pickle_file)
    rec = pd.DataFrame(rec)

print('Script: 12.00.03 [Load Table into Memory] completed')
end = timer()
print(end - start, 'seconds')

# =============================================================================
# 12.00.04 | Review
# =============================================================================
start = timer()
#print(rec.head())
#print(rec.info())

print('Script: 12.00.03 [Review Predictions] completed')
end = timer()
print(end - start, 'seconds')

# =============================================================================
# 12.01.01 | Read in predictions - cluster
# =============================================================================
start = timer()

c_frame = pd.read_csv("./output/clusters/reviewer/camera/camera_reviewer_cluster.csv")
print('Number of Rows:'+ str(len(c_frame)))

print('Script: 12.01.01 [Cluster Predictions] completed')
end = timer()
print(end - start, 'seconds')

# =============================================================================
# 12.01.02 | Melt cluster labels
# =============================================================================
start = timer()

m_frame = pd.melt(c_frame, id_vars='labels')
print('Number of Rows:'+ str(len(m_frame)))

m_frame = m_frame.dropna(subset=['value'])
print('Number of Rows:'+ str(len(m_frame)))

print('Script: 12.01.02 [Melt] completed')
end = timer()
print(end - start, 'seconds')

# =============================================================================
# 12.01.03 | Filter Clusters
# =============================================================================
start = timer()

m_frame = m_frame[m_frame['variable']=='0']
m_frame.rename(columns={'variable':'reviewer_cluster'}, inplace=True)
m_frame = m_frame.drop(columns='value')

print('Number of Rows:'+ str(len(m_frame)))

print(m_frame.info())

print('Script: 12.01.03 [Filter Clusters] completed')
end = timer()
print(end - start, 'seconds')

# =============================================================================
# 12.02.01 | Read Reviews Data
# =============================================================================
start = timer()

with open((working_directory + '/data/pickles/og_pickles/reviewsdata_5.pkl'), 'rb') as pickle_file:
    review_data = pickle.load(pickle_file)
    review_data = pd.DataFrame(review_data)

print(review_data.info())

print('Script: 12.02.01 [Review Data] completed')
end = timer()
print(end - start, 'seconds')

# =============================================================================
# 12.03.01 | Label review data with clusters
# =============================================================================
start = timer()

review_data_c = pd.merge(left=review_data, right=m_frame, left_on='reviewerID', right_on='labels',
                         how='inner')

review_data_c = review_data_c.drop(columns='labels')

print(review_data_c.head())

print('Script: 12.03.01 [Review Data] completed')
end = timer()
print(end - start, 'seconds')

# =============================================================================
# 12.04.01 | Parameters for Top N
# =============================================================================
start = timer()

a = pd.pivot_table(review_data_c, index=['asin'], values=['overall'],
             aggfunc={'overall': [len, np.min, np.mean, np.median, np.max, np.sum]})

def conv_pivot2df(pivot_name):
    '''Pandas pivot table to data frame; requires pandas'''
    pivot_name.columns = pivot_name.columns.droplevel(0)
    pivot_name = pivot_name.reset_index().rename_axis(None, axis=1)
    pivot_name = pd.DataFrame(pivot_name)

conv_pivot2df(a)

# Filter so that only products with 5 or more reviews are considered
b = a[a['len']>=5]

# Sort the reviews
c = b['mean'].sort_values(ascending=False)
c = c.reset_index()

print('Script: 12.04.01 [Identify top products] completed')
end = timer()
print(end - start, 'seconds')

# =============================================================================
# 12.04.02 | Top N from Cluster 0
# =============================================================================
start = timer()
n = 10

d = c[0:n]

print(d)

d.to_csv("./output/clusters/rec/cluster0.csv")

print('Script: 12.04.02 [Review Data] completed')
end = timer()
print(end - start, 'seconds')
# ===============================================================================
# 13.00.01 | Ensemble | Documentation
# ===============================================================================
# Name:               13_Ensemble
# Author:             Kiley
# Last Edited Date:   11/17/19
# Description:        Ensemble of tools
#
# Notes:
#
#
# Warnings:
#
#
# Outline:
#
#
# =============================================================================
# 13.00.02 | Import Packages
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

print('Script: 13.00.02 [Import packages] completed')

# =============================================================================
# 13.00.03 | Readin predictions - SVD
# =============================================================================
start = timer()
cluster = clusters[0]

with open((working_directory + '/output' + '/baseline_model/svd_recommendations_cluster'+cluster+'.pkl'), 'rb') as pickle_file:
    rec = pickle.load(pickle_file)
    rec = pd.DataFrame(rec)

print('Script: 13.00.03 [Load Table into Memory] completed')
end = timer()
print(end - start, 'seconds')

# =============================================================================
# 13.00.04 | Review
# =============================================================================
start = timer()
#print(rec.head())
#print(rec.info())

print('Script: 13.00.04 [Review Predictions] completed')
end = timer()
print(end - start, 'seconds')

# =============================================================================
# 13.01.01 | Read in predictions - cluster
# =============================================================================
start = timer()

top_n_cluster = pd.read_csv("./output/clusters/rec/cluster0_all.csv")
print('Number of Rows:'+ str(len(top_n_cluster)))

print('Script: 13.01.01 [Cluster Predictions] completed')
end = timer()
print(end - start, 'seconds')

# =============================================================================
# 13.02.01 | Merge
# =============================================================================
start = timer()

print('Number of Rows [rec]:'+ str(len(rec)))
print('Number of Rows [top]:'+ str(len(top_n_cluster)))

df = pd.merge(left=rec, right=top_n_cluster, left_on='asin', right_on='asin',
              how='outer', indicator=True)

# Gut check
print(pd.pivot_table(df, index=['_merge'], values=['mean'], aggfunc=len))

print('Number of Rows:'+ str(len(df)))

print('Script: 13.02.01 [Merge] completed')
end = timer()
print(end - start, 'seconds')

# =============================================================================
# 13.03.01 | Calculate New Score
# =============================================================================
start = timer()

df['ensemble'] = (df['predicted_rating']+df['mean'])/2
df.to_csv("./output/ensemble/cluster0_all.csv")

print('Script: 13.03.01 [New Ensemble Score Creation] completed')
end = timer()
print(end - start, 'seconds')

# =============================================================================
# 13.04.01 | Top N
# =============================================================================
start = timer()

c = df.sort_values(by='ensemble', ascending=False)
n = 10

d = c[0:n]

d.to_csv("./output/ensemble/cluster0.csv")
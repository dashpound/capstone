# ===============================================================================
# 12.00.01 | Ensemble | Documentation
# ===============================================================================
# Name:               12_baseline_ratings_model_clusters
# Author:             Kiley
# Last Edited Date:   11/17/19
# Description:        Generate predictions for unrated items using SVD model.
#
# Notes:              Only generating predictions for SVD model since that was the 'best' model.
#                     Data formatting is a large part of this script since matrix format does not work for surprise.
#                     Therefore, need to get reviewer & product combinations into a data frame.
#                     Surprise also uses different ids to generate predictions so translation back to original ids is required.
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

from code.configuration.environment_configuration import RANDOM_SEED, working_directory

clusters=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

print('Script: 12.00.02 [Import packages] completed')

# =============================================================================
# 12.00.03 | Readin predictions
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
print(rec.head())
print(rec.info())

print('Script: 12.00.03 [Review Predictions] completed')
end = timer()
print(end - start, 'seconds')

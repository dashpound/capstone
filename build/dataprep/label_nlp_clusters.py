# ===============================================================================
# 09.00.01 | Label Clusters | Documentation
# ===============================================================================
# Name:               09_label_nlp_clusters
# Author:             Kiley
# Last Edited Date:   11/16/19
# Description:        This script adds user cluster to the review pickle
#
# Notes:
#
# Warnings:           The data checks are not automated right now but
#                     they are a starting point.
# Outline:
#                    Load data
#                    Load clusters
#                    Merge
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

print('Script: 09.00.02 [Import Packages] completed')

# =============================================================================
# 09.01.01 | Load Reviews Data
# =============================================================================
# Unpack review data where reviewer has left at least 5 reviews

with open(Path(working_directory+'/data'+reviews_path), 'rb') as pickle_file:
    reviews_df = pickle.load(pickle_file)
    reviews_df = pd.DataFrame(reviews_df)

reviews_df.drop(['user_cluster', 'user_cluster_camera_x', 'user_cluster_camera_y'], axis=1, inplace =True)

print('Script: 09.01.01 [Load Reviews Data] completed')

# =============================================================================
# 09.01.02 | Load Clusters
# =============================================================================
a = pd.read_csv(Path(working_directory+'/output/clusters/reviewer/camera/camera_reviewer_cluster.csv'), index_col='labels')

print('Script: 09.01.01 [Load Clusters] completed')

# =============================================================================
# 09.01.03 | Melt Clusters
# =============================================================================
b = pd.melt(a.reset_index(), id_vars='labels')

print('Script: 09.01.03 [Melt Clusters] completed')
# =============================================================================
# 09.01.04 | Filter Clusters
# =============================================================================
c = b[b['value']==1]
print('Script: 09.01.04 [Filter Clusters] completed')
# =============================================================================
# 09.01.05 | Rename Columns
# =============================================================================

c.columns = ['reviewerID', 'user_cluster_camera', 'indicator']
print('Script: 09.01.05 [Rename Columns] completed')

# =============================================================================
# 09.01.06 | Drop indicator column
# =============================================================================

c.drop(['indicator'], axis=1, inplace =True)
print('Script: 09.01.04 [Drop Column] completed')

# =============================================================================
# 09.01.03 | Merge Table
# =============================================================================
d = pd.merge(reviews_df, c, how="left", left_on='reviewerID', right_on='reviewerID')
d.to_pickle(Path(working_directory+'/data/pickles/enhanced/reviews_meta_combined_individual.pkl'))

print('Script: 09.01.07 [Merge Tables] completed')
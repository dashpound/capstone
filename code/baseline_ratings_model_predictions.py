# ===============================================================================
# 07.00.01 | Baseline Ratings Model Predictions | Documentation
# ===============================================================================
# Name:               07_baseline_ratings_model
# Author:             Rodd
# Last Edited Date:   11/9/19
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
#                     Prepare data for modeling by filtering the data to Camera & Photo.
#                     Transform data so that reviewer & product combinations are into a data frame.
#                     Filter data frame to only include unnrated items.
#                     Generate predictions for all unnrated items and get into format with original ids.
#                     Select the top n recommendations.
#                     Pickle recommendations to output folder.
#
#
# =============================================================================
# 07.00.02 | Import Packages
# =============================================================================
# Import packages
import pandas as pd
import numpy as np
import random
import gc
import pickle
from pathlib import Path

# Surprise functions
# pip install scikit-surprise
from surprise import Reader, Dataset
from surprise import dump

# Import modules (other scripts)
from data_load import reviews_df
from environment_configuration import RANDOM_SEED, working_directory

print('Script: 07.00.02 [Import packages] completed')


# =============================================================================
# 07.00.03 | Set seed and recommendation var
# =============================================================================
# setting seed this way, per surprise documentation
# https://surprise.readthedocs.io/en/stable/FAQ.html
RANDOM_SEED
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# setting this global param
number_of_recs = 10

print('Script: 07.00.03 [Set seed and recommendation var] completed')


# =============================================================================
# 07.01.01 | Load models
# =============================================================================
_, svd_algo = dump.load(Path(working_directory + '/output' + '/baseline_model/svd_model'))
#_, knn_algo = dump.load(Path(working_directory + '/output' + '/baseline_model/knn_model'))

print('Script: 07.01.01 [SVD model loaded] completed')


# =============================================================================
# 07.02.01 | Create data frame for unrated items
# =============================================================================
# goal is to generate predictions for items that users have not rated
# therefore, we have to get a data frame that contains items that users have not rated
reviews_df2 = reviews_df[reviews_df['category2_t']=="Camera & Photo"]

# select the columns we want
reviews_sub = reviews_df2[['reviewerID', 'asin', 'overall']]

# build a matrix with all products & reviewers
# selecting a subset of data to avoid mem issues
# small_data = reviews_sub.sample(frac=0.01)

# reviewers are rows, products are columns
data_pivot = reviews_sub.groupby(['reviewerID', 'asin'])['overall'].max().unstack()
gc.collect()
data_pivot.reset_index(inplace=True)

# have to gather the data set to get into a columnar format for surprise
gc.collect()
final_df = pd.melt(data_pivot,id_vars='reviewerID',var_name='asin',value_name='overall')
# now filter to only get products that haven't been rated by users
# equivalent to saying 'overall' is null
final_df = final_df[final_df['overall'].isnull()]
gc.collect()

# format data for surprise
ratings_dict = {'asin': list(final_df.asin),
                'reviewerID': list(final_df.reviewerID),
                'rating': list(final_df.overall)}

df = pd.DataFrame(ratings_dict) 

# A reader is still needed but only the rating_scale param is required.
# The Reader class is used to parse a file containing ratings.
reader = Reader(rating_scale=(1, 5))

# The columns must correspond to user id, item id and ratings (in that order).
data = Dataset.load_from_df(df[['reviewerID', 'asin', 'rating']], reader)

print('Script: 07.02.01 [Create ratings df for unrated items] completed')


# =============================================================================
# 07.03.01 | Generate predictions using SVD model
# =============================================================================
# SVD model
# it takes some time to fit and generate predictions
gc.collect()
trainset = data.build_full_trainset()
svd_algo.fit(trainset)
gc.collect()

svd_full_predictions = svd_algo.test(trainset.build_testset())
gc.collect()

print('Script: 07.03.01 [Generate predictions using SVD model] completed')


# =============================================================================
# 07.04.01 | Get top 10 recommendations
# =============================================================================
# https://towardsdatascience.com/building-and-testing-recommender-systems-with-surprise-step-by-step-d4ba702ef80b
def get_Iu(uid):
    """ return the number of items rated by given user
    args: 
      uid: the id of the user
    returns: 
      the number of items rated by the user
    """
    try:
        return len(trainset.ur[trainset.to_inner_uid(uid)])
    except ValueError: # user was not part of the trainset
        return 0
    
def get_Ui(iid):
    """ return number of users that have rated given item
    args:
      iid: the raw id of the item
    returns:
      the number of users that have rated the item.
    """
    try: 
        return len(trainset.ir[trainset.to_inner_iid(iid)])
    except ValueError:
        return 0

# first create a data frame with predictions for ALL unrated items for a user
svd_df = pd.DataFrame(svd_full_predictions, columns=['uid', 'iid', 'rui', 'est', 'details'])
svd_df.rename(columns={'uid':'reviewerID','iid':'asin','est':'predicted_rating'},inplace=True)
# can't deselect the columns above - otherwise get an error
svd_df = svd_df.drop(['rui','details'], axis=1)

# generate top 10 recommendations for each user
gc.collect()
svd_recs_df = svd_df.sort_values('predicted_rating',ascending = False).groupby('reviewerID').head(number_of_recs)

print('Script: 07.04.01 [Get top 10 recommendations] completed')


# =============================================================================
# 07.05.01 | Pickle SVD top 10 recommendations
# =============================================================================
svd_recs_df.to_pickle(Path(working_directory + '/output' + '/baseline_model/svd_recommendations.pkl'))

print('Script: 07.05.01 [Pickle SVD top 10 recommendations] completed')
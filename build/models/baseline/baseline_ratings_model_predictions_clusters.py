# ===============================================================================
# 11.00.01 | Baseline Ratings Model Predictions | Documentation
# ===============================================================================
# Name:               11_baseline_ratings_model_clusters
# Author:             Kiley
# Last Edited Date:   11/16/19
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
# 11.00.02 | Import Packages
# =============================================================================
# Import packages
import pandas as pd
import numpy as np
import random
import gc
import pickle
from pathlib import Path
from timeit import default_timer as timer

# Surprise functions
# pip install scikit-surprise
from surprise import Reader, Dataset
from surprise import dump

# Import modules (other scripts)
from code.dataprep.data_load import reviews_df
from code.configuration.environment_configuration import RANDOM_SEED, working_directory

print('Script: 11.00.02 [Import packages] completed')

# =============================================================================
# 11.00.03 | Set seed and recommendation var
# =============================================================================
start = timer()
# setting seed this way, per surprise documentation
# https://surprise.readthedocs.io/en/stable/FAQ.html
RANDOM_SEED
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# setting this global param
number_of_recs = 10
clusters=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

print('Script: 11.00.03 [Set seed and recommendation var] completed')
end = timer()
print(end - start, 'seconds')

# =============================================================================
# 11.01.01 | Load models
# =============================================================================
for cluster in clusters:
    print('Now Starting Cluster... ', cluster)
    gc.collect()
    start = timer()
    _, svd_algo = dump.load(Path(working_directory + '/output' + '/baseline_model/svd_model_cluster'+cluster))
    #_, knn_algo = dump.load(Path(working_directory + '/output' + '/baseline_model/knn_model'))
    gc.collect()

    print('Script: 11.01.01 [SVD model loaded] completed')
    end = timer()
    print(end - start, 'seconds')

    # =============================================================================
    # 11.02.01 | Create data frame for unrated items
    # =============================================================================
    start = timer()
    # goal is to generate predictions for items that users have not rated
    # therefore, we have to get a data frame that contains items that users have not rated
    reviews_df2 = reviews_df[reviews_df['category2_t']=="Camera & Photo"]
    gc.collect()
    print('Script: 11.02.01 [Filtered Camera & Photo] completed')
    end = timer()
    print(end - start, 'seconds')

    # =============================================================================
    # 11.02.02 | Filtered to Cluster
    # =============================================================================
    start = timer()
    reviews_df2 = reviews_df[reviews_df['user_cluster_camera'] == cluster]
    gc.collect()
    print('Script: 11.02.02 [Filtered to cluster] completed')
    end = timer()
    print(end - start, 'seconds')

    # =============================================================================
    # 11.02.03 | Dropped Columns
    # =============================================================================
    start = timer()

    # select the columns we want
    reviews_sub = reviews_df2[['reviewerID', 'asin', 'overall']]
    gc.collect()
    print('Script: 11.02.03 [Dropped columns] completed')
    end = timer()
    print(end - start, 'seconds')

    # =============================================================================
    # 11.02.04 | Move it Around
    # =============================================================================
    start = timer()
    # build a matrix with all products & reviewers
    # selecting a subset of data to avoid mem issues
    # small_data = reviews_sub.sample(frac=0.01)
    # reviewers are rows, products are columns
    data_pivot = reviews_sub.groupby(['reviewerID', 'asin'])['overall'].max().unstack()
    gc.collect()
    print('Script: 11.02.04 [Move it around] completed')
    end = timer()
    print(end - start, 'seconds')

    # =============================================================================
    # 11.02.05 | Reset Index
    # =============================================================================
    start = timer()

    data_pivot.reset_index(inplace=True)
    gc.collect()
    print('Script: 11.02.05 [Reset index] completed')
    end = timer()
    print(end - start, 'seconds')

    # =============================================================================
    # 11.02.06 | Melt Pivot
    # =============================================================================
    start = timer()

    # have to gather the data set to get into a columnar format for surprise
    final_df = pd.melt(data_pivot,id_vars='reviewerID',var_name='asin',value_name='overall')
    gc.collect()
    print('Script: 11.02.06 [Melt pivot] completed')
    end = timer()
    print(end - start, 'seconds')

    # =============================================================================
    # 11.02.07 | Filter to unrated products
    # =============================================================================
    start = timer()

    # now filter to only get products that haven't been rated by users
    # equivalent to saying 'overall' is null
    final_df = final_df[final_df['overall'].isnull()]
    gc.collect()
    print('Script: 11.02.07 [Filter to unrated products] completed')
    end = timer()
    print(end - start, 'seconds')

    # =============================================================================
    # 11.02.08 | Reformat Data
    # =============================================================================
    start = timer()

    # format data for surprise
    ratings_dict = {'asin': list(final_df.asin),
                    'reviewerID': list(final_df.reviewerID),
                    'rating': list(final_df.overall)}
    gc.collect()
    print('Script: 11.02.08 [Reformat Data] completed')
    end = timer()
    print(end - start, 'seconds')

    # =============================================================================
    # 11.02.09 | Convert to DF
    # =============================================================================
    start = timer()

    df = pd.DataFrame(ratings_dict)
    gc.collect()
    print('Script: 11.02.09 [Convert to DF] completed')
    end = timer()
    print(end - start, 'seconds')

    # =============================================================================
    # 11.02.10 | Create Reader
    # =============================================================================
    start = timer()

    # A reader is still needed but only the rating_scale param is required.
    # The Reader class is used to parse a file containing ratings.
    reader = Reader(rating_scale=(1, 5))
    gc.collect()
    print('Script: 11.02.10 [Create Reader] completed')
    end = timer()
    print(end - start, 'seconds')

    # =============================================================================
    # 11.02.11 | Create data frame for unrated items
    # =============================================================================
    start = timer()

    # The columns must correspond to user id, item id and ratings (in that order).
    data = Dataset.load_from_df(df[['reviewerID', 'asin', 'rating']], reader)
    gc.collect()

    print('Script: 11.02.11 [Pickle SVD top 10 recommendations] completed')
    end = timer()
    print(end - start, 'seconds')

    # =============================================================================
    # 11.03.01 | Generate predictions using SVD model
    # =============================================================================
    start = timer()
    # SVD model
    # it takes some time to fit and generate predictions

    trainset = data.build_full_trainset()
    gc.collect()

    print('Script: 11.03.01 [Build Trainset] completed')
    end = timer()
    print(end - start, 'seconds')

    # =============================================================================
    # 11.03.02 | Fit Trainset
    # =============================================================================
    start = timer()

    svd_algo.fit(trainset)
    gc.collect()

    print('Script: 11.03.02 [Fit Trainset] completed')
    end = timer()
    print(end - start, 'seconds')

    # =============================================================================
    # 11.03.03 | Full predictions
    # =============================================================================
    start = timer()

    svd_full_predictions = svd_algo.test(trainset.build_testset())
    gc.collect()

    print('Script: 11.03.03 [Full predictions] completed')
    end = timer()
    print(end - start, 'seconds')

    # =============================================================================
    # 11.04.01 | Get top 10 recommendations
    # =============================================================================
    start = timer()
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

    print('Script: 11.04.01 [Define functions] completed')
    end = timer()
    print(end - start, 'seconds')
    gc.collect()


    # first create a data frame with predictions for ALL unrated items for a user
    # =============================================================================
    # 11.04.02 | Create a dataframe of predictions
    # =============================================================================
    start = timer()

    svd_df = pd.DataFrame(svd_full_predictions, columns=['uid', 'iid', 'rui', 'est', 'details'])
    gc.collect()

    print('Script: 11.04.02 [Dataframe of predictions] completed')
    end = timer()
    print(end - start, 'seconds')

    # =============================================================================
    # 11.04.03 | Relabel
    # =============================================================================
    start = timer()

    svd_df.rename(columns={'uid':'reviewerID','iid':'asin','est':'predicted_rating'},inplace=True)
    gc.collect()

    print('Script: 11.04.03 [Pickle SVD top 10 recommendations] completed')
    end = timer()
    print(end - start, 'seconds')

    # can't deselect the columns above - otherwise get an error
    # =============================================================================
    # 11.04.04 | Drop Unused Columns
    # =============================================================================
    start = timer()

    svd_df = svd_df.drop(['rui','details'], axis=1)
    gc.collect()

    print('Script: 11.04.04 [Drop unused columns] completed')
    end = timer()
    print(end - start, 'seconds')

    # =============================================================================
    # 11.04.05 | Sort ratings
    # =============================================================================
    start = timer()

    # generate top 10 recommendations for each user
    svd_recs_df = svd_df.sort_values('predicted_rating',ascending = False).groupby('reviewerID').head(number_of_recs)
    gc.collect()

    print('Script: 11.04.05 [Sort predictions] completed')
    end = timer()
    print(end - start, 'seconds')

    # =============================================================================
    # 11.05.01 | Pickle SVD top 10 recommendations
    # =============================================================================
    start = timer()
    svd_recs_df.to_pickle(Path(working_directory + '/output' + '/baseline_model/svd_recommendations_cluster'+cluster+'.pkl'))
    gc.collect()

    print('Script: 11.05.01 [Pickle SVD top 10 recommendations] completed')
    end = timer()
    print(end - start, 'seconds')
print('Script 11 Complete')
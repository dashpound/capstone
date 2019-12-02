# ===============================================================================
# 10.00.01 | Baseline by Cluster | Documentation
# ===============================================================================
# Name:               10_baseline_ratings_model
# Author:             Kiley
# Last Edited Date:   11/16/19
# Description:        Build SVD & KNN models from the surprise package using only ratings data.
#                     
# Notes:              Used the raw ratings data, but KNNMeans algorithm was able to account for average ratings of users.
#                     Results use the Camera & Photo data.
#                     Model methods were selected by reading surprise documentation.
#
# Warnings:
#
#
# Outline:            Imports needed packages and set seed for reproducibility.
#                     Prepare data for modeling by filtering the data to Camera & Photo and pulling out 3 vars.
#                     Split data into train/test sets using 80/20 split.
#                     Run a grid search to determine the best params for an SVD model.
#                     Calculate GOF statistics for SVD model.
#                     Save SVD model in output folder.
#                     Run a grid search to determine the best params for the KNN model.
#                     Calculate GOF statistics for KNN model.
#                     Save KNN model in output folder.
#
#
# =============================================================================
# 10.00.02 | Import packages
# =============================================================================
# Import packages
import pandas as pd
import random
import numpy as np
from pathlib import Path
from timeit import default_timer as timer
import gc

# Surprise functions
# pip install scikit-surprise
from surprise import Reader, Dataset, SVD, KNNWithMeans
from surprise.model_selection import cross_validate, train_test_split, GridSearchCV
from surprise.accuracy import rmse, mae, fcp
from surprise import dump

# Import modules (other scripts)
from code.dataprep.data_load import reviews_df
from code.configuration.environment_configuration import RANDOM_SEED, working_directory, modeling_path

print('Script: 10.00.02 [Import packages] completed')

# =============================================================================
# 10.00.03 | Set Seed
# =============================================================================
# setting seed this way, per surprise documentation
# https://surprise.readthedocs.io/en/stable/FAQ.html
start = timer()

RANDOM_SEED
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
clusters=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

print('Script: 10.00.03 [Set seed] completed')
end = timer()
print(end - start, 'seconds')

# =============================================================================
# 10.01.01 | Prepare data for modeling
# =============================================================================
start = timer()
# https://github.com/khanhnamle1994/movielens/blob/master/Content_Based_and_Collaborative_Filtering_Models.ipynb
# filter the data to only include camera & photo
# NOTE: this changed b/c Hemant overwrote the original reviews data with his cleaned reviews data
# this data frame has the products fields
for cluster in clusters:
    print('Now Starting Cluster... ', cluster)
    gc.collect()
    reviews_df2 = reviews_df[reviews_df['category2_t']=="Camera & Photo"]
    reviews_df2 = reviews_df[reviews_df['user_cluster_camera']==cluster]

    # select the columns we want
    reviews_sub = reviews_df2[['reviewerID', 'asin', 'overall']]

    # summary data frames - for reference
    reviewers_pivot = reviews_sub.groupby(['reviewerID']).size().reset_index().rename(columns = {0: 'reviews'})[['reviewerID','reviews']]
    # most have a single review
    reviews_by_reviewers = reviewers_pivot.groupby(['reviews']).aggregate({'reviewerID': pd.Series.nunique}).reset_index()
    # min is still 5 reviews
    products_pivot = reviews_sub.groupby(['asin']).size().reset_index().rename(columns = {0: 'reviews'})[['asin','reviews']]
    reviews_by_products = products_pivot.groupby(['reviews']).aggregate({'asin': pd.Series.nunique}).reset_index()

    print('Script: 10.01.01 [Initial data frame created] completed')
    end = timer()
    print(end - start, 'seconds')

    # =============================================================================
    # 10.02.01 | Create train/test data frames
    # =============================================================================
    start = timer()
    # https://towardsdatascience.com/building-and-testing-recommender-systems-with-surprise-step-by-step-d4ba702ef80b
    ratings_dict = {'asin': list(reviews_sub.asin),
                    'reviewerID': list(reviews_sub.reviewerID),
                    'rating': list(reviews_sub.overall)}

    df = pd.DataFrame(ratings_dict)

    # A reader is still needed but only the rating_scale param is required.
    # The Reader class is used to parse a file containing ratings.
    reader = Reader(rating_scale=(1, 5))

    # The columns must correspond to user id, item id and ratings (in that order).
    data = Dataset.load_from_df(df[['reviewerID', 'asin', 'rating']], reader)

    # split into train & test sets using 80/20 split
    trainset, testset = train_test_split(data, test_size=0.20)

    print('Script: 10.02.01 [Train and test sets created] completed')
    end = timer()
    print(end - start, 'seconds')

    # =============================================================================
    # 10.03.01 | Optimize SVD model
    # =============================================================================
    start = timer()
    #gc.collect()
    #param_grid = {
    #    "n_epochs": [5, 10, 20],
    #    "lr_all": [0.01, 0.001, .005], #.005 is default
    #    "reg_all": [0.1, 0.5, .02, .001] #.02 is default
    #}
    #
    ## use 3-fold CV as that significantly speeds up fitting over 5-folds
    #gs = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"], cv=3)
    #
    #gs.fit(data)
    #
    #print('Script: 06.03.01 [Grid search for SVD model] completed')
    #
    #print(gs.best_score["rmse"])
    #print(gs.best_params["rmse"])

    # create a model with the best params according to rmse
    # svd_algo = gs.best_estimator['rmse']
    svd_algo = SVD(n_epochs=20, lr_all=0.01, reg_all=0.5, random_state=RANDOM_SEED)
    svd_predictions = svd_algo.fit(trainset).test(testset)

    # evaluate model
    rmse(svd_predictions)
    mae(svd_predictions)
    fcp(svd_predictions)

    # cross validation results
    # this outputs a nice table if verbose=True
    svd_cv_output = cross_validate(svd_algo, data, measures=["rmse", "mae", "fcp"], cv=3, verbose=True)

    print('Script: 10.03.01 [SVD model and initial GOF statistics] completed')
    end = timer()
    print(end - start, 'seconds')

    # =============================================================================
    # 10.03.02 | Compute precision & recall at k for SVD model
    # =============================================================================
    start = timer()
    # https://surprise.readthedocs.io/en/stable/FAQ.html
    from collections import defaultdict
    from surprise.model_selection import KFold

    def precision_recall_at_k(predictions, k=10, threshold=3.5):
        '''Return precision and recall at k metrics for each user.'''

        # First map the predictions to each user.
        user_est_true = defaultdict(list)
        for uid, _, true_r, est, _ in predictions:
            user_est_true[uid].append((est, true_r))

        precisions = dict()
        recalls = dict()
        for uid, user_ratings in user_est_true.items():

            # Sort user ratings by estimated value
            user_ratings.sort(key=lambda x: x[0], reverse=True)

            # Number of relevant items
            n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

            # Number of recommended items in top k
            n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

            # Number of relevant and recommended items in top k
            n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                                  for (est, true_r) in user_ratings[:k])

            # Precision@K: Proportion of recommended items that are relevant
            precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

            # Recall@K: Proportion of relevant items that are recommended
            recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

        return precisions, recalls

    # define our inputs
    kf = KFold(n_splits=3,random_state=RANDOM_SEED)
    algo = svd_algo

    for trainset, testset in kf.split(data):
        algo.fit(trainset)
        predictions = algo.test(testset)
        precisions, recalls = precision_recall_at_k(predictions, k=3, threshold=4.5)
        # Precision and recall can then be averaged over all users
        print(sum(prec for prec in precisions.values()) / len(precisions))
        print(sum(rec for rec in recalls.values()) / len(recalls))

    # then mean and std values were calculated just manually using an approach like below
    # np.std(np.array([(0.8863066425613811,0.8881494827385794,0.8839016221058171)]))

    print('Script: 10.03.02 [Remaining GOF statistics for SVD model] completed')
    end = timer()
    print(end - start, 'seconds')

    # =============================================================================
    # 10.04.01 | Save SVD Model
    # =============================================================================
    start = timer()
    dump.dump(Path(working_directory + '/output' + '/baseline_model/svd_model_cluster'+cluster), algo=svd_algo)

    print('Script: 10.04.01 [Save SVD Model] completed')
    end = timer()
    print(end - start, 'seconds')

    # =============================================================================
    # 06.05.01 | Optimize KNNWithMeans model - Item Filtering
    # =============================================================================
    start = timer()
    # have to set up the param grid just a little bit differently for this model
    # choosing KNNWithMeans b/c it takes into account the mean ratings of each user
    # this takes some time to run b/c testing a lot of different combinations...
    #gc.collect()
    #sim_options = {
    #    "name": ["msd", "cosine", "pearson"],
    #    "min_support": [3, 4, 5],
    #    'user_based': [False]  # we want similarities between items
    #}
    #
    #bsl_options = {"n_epochs": [5, 10, 20]}
    #
    #param_grid = {"sim_options": sim_options,
    #              "k": [20, 40, 60], #  The (max) number of neighbors to take into account for aggregation. Default is 40.
    #              "bsl_options": bsl_options}
    #
    #gs = GridSearchCV(KNNWithMeans, param_grid, measures=["rmse", "mae"], cv=3)
    #gs.fit(data)
    #
    #print('Script: 06.04.01 [Grid search for KNN model] completed')
    #
    #print(gs.best_score["rmse"])
    #print(gs.best_params["rmse"])

    # build model with best parameters
    # knn_algo = gs.best_estimator['rmse']
    sim_options = {'name': 'pearson', 'min_support': 5, 'user_based': False}
    bsl_options = {"n_epochs": 5}

    knn_algo = KNNWithMeans(sim_options=sim_options,k=20,bsl_options=bsl_options,random_state=RANDOM_SEED)
    knn_predictions = knn_algo.fit(trainset).test(testset)

    # evaluate model
    rmse(knn_predictions)
    mae(knn_predictions)
    fcp(knn_predictions)

    # cross validation results
    knn_cv_output = cross_validate(knn_algo, data, measures=["rmse", "mae", "fcp"], cv=3, verbose=True)
    # https://bmanohar16.github.io/blog/recsys-evaluation-in-surprise - some nice code for plotting results

    print('Script: 10.05.01 [KNN model and initial GOF statistics] completed')
    end = timer()
    print(end - start, 'seconds')

    # =============================================================================
    # 10.05.02 | Precision and recall at k - KNN
    # =============================================================================
    start = timer()
    # define our inputs
    kf = KFold(n_splits=3,random_state=RANDOM_SEED)
    algo = knn_algo

    for trainset, testset in kf.split(data):
        algo.fit(trainset)
        predictions = algo.test(testset)
        precisions, recalls = precision_recall_at_k(predictions, k=3, threshold=4.5)
        # Precision and recall can then be averaged over all users
        print(sum(prec for prec in precisions.values()) / len(precisions))
        print(sum(rec for rec in recalls.values()) / len(recalls))

    # then mean and std values were calculated just manually using an approach like below
    # np.std(np.array([(0.8863066425613811,0.8881494827385794,0.8839016221058171)]))

    print('Script: 10.05.02 [Remaining GOF statistics for KNN model] completed')
    end = timer()
    print(end - start, 'seconds')

    # =============================================================================
    # 06.06.01 | save KNN Model
    # =============================================================================
    start = timer()
    dump.dump(Path(working_directory + '/output' + '/baseline_model/knn_model_cluster'+cluster), algo=knn_algo)

    print('Script: 10.06.01 [Saved KNN Model] completed')
    end = timer()
    print(end - start, 'seconds')
print('Script 10 Completed')
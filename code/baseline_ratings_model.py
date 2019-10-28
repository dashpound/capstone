# Import packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from math import sqrt
import gc

# Import modules (other scripts)
from clean_data_load import products_clean
from data_load import reviews_df
from functions import conv_pivot2df
from environment_configuration import RANDOM_SEED


# =============================================================================
# 05.01.01 | Create train/test data frames
# =============================================================================
# https://github.com/khanhnamle1994/movielens/blob/master/Content_Based_and_Collaborative_Filtering_Models.ipynb
reviews_df2 = pd.merge(products_clean[['title','asin']], reviews_df, on='asin',how='inner')

# have to change data types
reviews_df2['overall'] = reviews_df2['overall'].astype(int)

# transforming existing reviewer and product ids into int vars
# see that in calculating similarity, these vars are integer
unique_reviewers = pd.DataFrame(reviews_df2['reviewerID'].unique())
new_reviewers = pd.DataFrame(list(range(1, unique_reviewers.shape[0]+1)))
reviewers_new_mapping = pd.concat([unique_reviewers, new_reviewers], axis=1)
reviewers_new_mapping.columns = ['reviewerID', 'new_reviewerID']
#reviewers_new_mapping['new_reviewerID'] = reviewers_new_mapping['new_reviewerID'].astype(int)

unique_products = pd.DataFrame(reviews_df2['asin'].unique())
new_products = pd.DataFrame(list(range(192403, unique_products.shape[0]+192403)))
products_new_mapping = pd.concat([unique_products, new_products], axis=1)
products_new_mapping.columns = ['asin', 'new_asin']
#products_new_mapping['new_asin'] = products_new_mapping['new_asin'].astype(int)

# have to join those back to the original data frame
reviews_df2 = pd.merge(reviewers_new_mapping, reviews_df2, on='reviewerID',how='inner')
reviews_df2 = pd.merge(products_new_mapping, reviews_df2, on='asin',how='inner')

# also want to calculate reviewer's average rating in order to compute adjusted cosine similarity
# calculates avg reviewer rating and subtracts that from a user's score
avg_rating = reviews_df2[['reviewerID','overall']].groupby(['reviewerID'], as_index = False, sort = False).mean().rename(columns = {'overall': 'rating_mean'})[['reviewerID','rating_mean']]

reviews_df2 = pd.merge(reviews_df2,avg_rating,on = 'reviewerID', how = 'left')
reviews_df2['rating_adjusted'] = reviews_df2['overall']-reviews_df2['rating_mean']

# repeat for product
avg_rating2 = reviews_df2[['asin','overall']].groupby(['asin'], as_index = False, sort = False).mean().rename(columns = {'overall': 'product_rating_mean'})[['asin','product_rating_mean']]

reviews_df2 = pd.merge(reviews_df2,avg_rating2,on = 'asin', how = 'left')
reviews_df2['product_rating_adjusted'] = reviews_df2['overall']-reviews_df2['product_rating_mean']

# select the columns we want
reviews_sub = reviews_df2[['new_reviewerID', 'new_asin', 'rating_adjusted']]

# Randomly sample 1% of the ratings
# This is just to make sure the code works
small_data = reviews_sub.sample(frac=0.01)
# Check the sample info
print(small_data.info())

# split into training & testing sets
# use an 80/20 split
gc.collect()
train_data, test_data = train_test_split(small_data, test_size=0.2, random_state = RANDOM_SEED, shuffle=True)

# transform into matrix format
train_data_matrix = np.matrix(train_data)
test_data_matrix = np.matrix(test_data)

# check their shape
print(train_data_matrix.shape)
print(test_data_matrix.shape)


# =============================================================================
# 05.02.01 | Calculate item similarity - Pearson Correlation Coefficient
# =============================================================================
gc.collect()

# Item Similarity Matrix
item_correlation = 1 - pairwise_distances(train_data_matrix.T, metric='correlation')
item_correlation[np.isnan(item_correlation)] = 0 # have to set missing ratings to 0
print(item_correlation[:4, :4])


# =============================================================================
# 05.02.02 | Generate predictions - Pearson
# =============================================================================
# Function to predict ratings
def predict(ratings, similarity):
    pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

# Function to calculate RMSE
def rmse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return sqrt(mean_squared_error(pred, actual))

# Predict ratings on the training data with both similarity score
item_prediction = predict(train_data_matrix, item_correlation)

# RMSE on the test data
print('Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix)))

# RMSE on the train data
print('Item-based CF RMSE: ' + str(rmse(item_prediction, train_data_matrix)))


# =============================================================================
# 05.03.01 | Data transformation
# =============================================================================
#from scipy.sparse import csr_matrix
## get ValueError: Unstacked DataFrame is too big, causing int32 overflow
## pivot is very memory intensive
## reviewers_products_pivot = small_data.pivot(index='new_asin',columns='new_reviewerID',values='rating_adjusted').fillna(1e-8)
#gc.collect()
#train_pivot = train_data.groupby(['new_asin', 'new_reviewerID'])['rating_adjusted'].max().unstack()
#
#gc.collect()
## want to fill missing values with a small values that is not 0
#train_pivot.fillna(1e-8, inplace = True)
#
## convert dataframe of movie features to scipy sparse matrix
#train_matrix = csr_matrix(train_pivot.values)
#

# =============================================================================
# 05.03.02 | Calculate item similarity - Cosine similarity
# =============================================================================
# there are several diff functions for cosine similarity...linear_kernel is supposed to be faster
from sklearn.metrics.pairwise import linear_kernel
import sklearn.metrics.pairwise as pw
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity

gc.collect()
# Item Similarity Matrix
item_cosine_sim = 1 - pairwise_distances(train_data_matrix.T, metric='cosine')
#item_cosine_sim[np.isnan(item_cosine_sim)] = 0 # have to set missing ratings to 0
print(item_cosine_sim[:4, :4])


# =============================================================================
# 05.03.03 | Generate predictions - Cosine similarity
# =============================================================================

# Predict ratings on the training data with both similarity score
item_cosine_prediction = predict(train_data_matrix, item_cosine_sim)

# RMSE on the test data
print('Item-based CF RMSE: ' + str(rmse(item_cosine_prediction, test_data_matrix)))

# RMSE on the train data
print('Item-based CF RMSE: ' + str(rmse(item_cosine_prediction, train_data_matrix)))\


# =============================================================================
# 05.04.01 | Return predictions
# =============================================================================
# THIS CODE HAS NOT BEEN TESTED!
# build a 1-dimensional array with product titles
titles = reviews_df2['title']
indices = pd.Series(reviews_df2['new_asin'], index=reviews_df2['title'])

# build a function to get recommendations
# returns Pearson correlation coefficient recommendations
# trying to pass in title....
def product_recommendations(asin):
    idx = indices[asin]
    sim_scores = list(enumerate(item_correlation[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]
    product_indices = [i[0] for i in sim_scores]
    return titles.iloc[product_indices]

product_recommendations('Rand McNally 528881469 7-inch Intelliroute TND 700 Truck GPS').head(20)


# =============================================================================
# 05.05.01 | Model-based approach using surprise package
# =============================================================================
# https://realpython.com/build-recommendation-engine-collaborative-filtering/#user-based-vs-item-based-collaborative-filtering
# https://towardsdatascience.com/building-and-testing-recommender-systems-with-surprise-step-by-step-d4ba702ef80b

# pip install scikit-surprise
from surprise import Reader, Dataset, SVD, KNNWithMeans
from surprise.model_selection import cross_validate, train_test_split, GridSearchCV
from surprise.accuracy import rmse

# need to redefine the subset here b/c want the raw rating - not the adjusted rating
# to load dataset from pandas df, we need `load_fromm_df` method in surprise lib
# select the columns we want
reviews_sub = reviews_df2[['reviewerID', 'asin', 'overall']]

# Randomly sample 1% of the ratings
# This is just to make sure the code works
small_data = reviews_sub.sample(frac=0.01)
# Check the sample info
print(small_data.info())

# split into training & testing sets
# use an 80/20 split

ratings_dict = {'asin': list(small_data.asin),
                'reviewerID': list(small_data.reviewerID),
                'rating': list(small_data.overall)}

df = pd.DataFrame(ratings_dict)

# A reader is still needed but only the rating_scale param is required.
# The Reader class is used to parse a file containing ratings.
reader = Reader(rating_scale=(1., 5.))

# The columns must correspond to user id, item id and ratings (in that order).
data = Dataset.load_from_df(df[['reviewerID', 'asin', 'rating']], reader)

# split into train & test sets using 80/20 split
trainset, testset = train_test_split(data, test_size=0.20)

#################### BASELINE
benchmark = []
# Iterate over all algorithms
for algorithm in [SVD(), KNNWithMeans()]:
    # Perform cross validation
    results = cross_validate(algorithm, data, measures=['RMSE'], cv=3, verbose=False)
    
    # Get results & append algorithm name
    tmp = pd.DataFrame.from_dict(results).mean(axis=0)
    tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
    benchmark.append(tmp)
    
pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse') 

################# KNN APPROACH
# KNNWithMeans is basic collaborative filtering algorithm, taking into account the mean ratings of each user.
# Are we really doing item-item collaborative filtering?
gc.collect()
sim_options = {
    "name": ["msd", "cosine"],
    "min_support": [3, 4, 5],
    "user_based": [False, True],
}

param_grid = {"sim_options": sim_options}

gs = GridSearchCV(KNNWithMeans, param_grid, measures=["rmse", "mae"], cv=3)
gs.fit(data)

print(gs.best_score["rmse"])
print(gs.best_params["rmse"])


knn_algo = KNNWithMeans(sim_options=gs.best_params["rmse"])
knn_predictions = knn_algo.fit(trainset).test(testset)
rmse(knn_predictions)
# cross validation
cross_validate(knn_algo, data, measures=['RMSE'], cv=3, verbose=False)

############################### SVD APPROACH/MATRIX FACTORIZATION
gc.collect()
param_grid = {
    "n_epochs": [5, 10],
    "lr_all": [0.002, 0.005],
    "reg_all": [0.4, 0.6]
}
gs = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"], cv=3)

gs.fit(data)

print(gs.best_score["rmse"])
print(gs.best_params["rmse"])

# not trusting call the best params
svd_algo = SVD(n_epochs = 10, lr_all = .005, reg_all = .4)
svd_predictions = svd_algo.fit(trainset).test(testset)
rmse(svd_predictions)
# cross validation
cross_validate(svd_algo, data, measures=['RMSE'], cv=3, verbose=False)

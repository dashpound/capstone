# ===============================================================================
# 08.00.01 | Predict ratings with Deep Neural Network Collaborative Filtering Model | Documentation
# ===============================================================================
# Name:               08_Predict_Collaborative_DNN_model
# Author:             Merrill
# Last Edited Date:   11/9/19
# Description:        Build DNN model using only user ratings and product data
#                     
# Notes:              Built model with both IDs only and full metadata; full 
#                     metadata model overfit data returning same similarity for
#                     all users
#
# Warnings:
#
#
# Outline:            Imports needed packages and set seed for reproducibility.
#                     Import dataframes for predictions
#                     Predict user ratings
#
#
# =============================================================================
# 08.00.02 | Import packages
# =============================================================================
# Import packages
#import pickle, datetime, os, gc
import pandas as pd
from compress_pickle import load
import random
import matplotlib.pyplot as plt
#from pathlib import Path
from tensorflow import keras
from tensorflow.keras import backend as K

# Import modules (other scripts)
from code.configuration.environment_configuration import *
#from clean_data_load import *
#from data_load import *

#console output flag
verbose = True
#create data files flag
create_data_files = False

#load dnn dataframes
merged_3 = load("./data/pickles/enhanced/dnn_merged_3.gz")
user_df = load("./data/pickles/enhanced/dnn_user_df.gz")
prod_df = load("./data/pickles/enhanced/dnn_prod_df.gz")
tfidf_df = load("./data/pickles/enhanced/dnn_tfidf_df.gz")
"""
merged_3 = pd.read_pickle("./data/pickles/enhanced/dnn_merged_3.pkl")
user_df = pd.read_pickle("./data/pickles/enhanced/dnn_user_df.pkl")
prod_df = pd.read_pickle("./data/pickles/enhanced/dnn_prod_df.pkl")
tfidf_df = pd.read_pickle("./data/pickles/enhanced/dnn_tfidf_df.pkl")
"""
products_clean = pd.read_pickle("./data/pickles/enhanced/dnn_products_clean_idx_all.pkl")
user2idx_df = pd.read_pickle("./data/pickles/enhanced/user2idx_20191114.pkl")

#this function needs to be stored in a seperate file since used in both scripts
#custom r2 metric
def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()))

#custom_objects
dependencies = {
    'r2_keras': r2_keras
}
#load dnn model
model = keras.models.load_model('./code/models/deep/models/merged_2_embedding_2L_6_3_64f_20191117_cat_nlp.h5',
                                custom_objects=dependencies)

print('Script: 08.00.02 [Import packages] completed')
# =============================================================================
# 08.01.01 | Create similarity function for unreviewed products by specified reviewer
# =============================================================================
def get_sim_products_by_user(rev_id):
    #look up reviewer index number
    u_test_idx = user2idx_df.loc[rev_id]
    
    #get all of the products this reviewer has not reviewed
    #test = products_clean[~products_clean.asin.isin(merged_2[merged_2['reviewerID']==rev_id].asin)].idx
    test = products_clean[~products_clean.asin.isin(merged_3[merged_3['reviewerID']==rev_id].asin)].index
    
    #setup test dataframe
    test = pd.DataFrame(test, columns=['prodIdx'])
    #test.rename(columns={"idx":'prodIdx'}, inplace=True)
    test['userIdx'] = u_test_idx.idx
    
    #setup product DF for predictions
    prod_avg = prod_df[['prodIdx', 'price_t', 'numberQuestions', 'numberReviews',
                        'meanStarRating', 'cat_idx']].groupby(by='prodIdx').mean()
    #prod_unq = prod_df.drop_duplicates()
    #prod_test_pop = pd.merge(prod_avg, prod_unq, how='inner', on='prodIdx')
    prod_test_user = pd.merge(test.prodIdx, prod_avg, how='inner', on='prodIdx')
    t=tfidf_df[tfidf_df.userIdx!=u_test_idx.idx].iloc[:,1:].drop_duplicates()
    prod_tfidf_user = pd.merge(test.prodIdx, t, how='inner', on='prodIdx')
    prod_tfidf_user.drop(columns=['prodIdx'], inplace=True)
    #setup user DF for predictions
    #u=user_df[user_df.userIdx==u_test_idx.idx].drop_duplicates()
    u=user_df[user_df.userIdx==u_test_idx.idx].groupby(by='userIdx').mean().reset_index()
    user_test_user = pd.concat([u]*prod_test_user.shape[0], ignore_index=True)
    [prod_test_user.shape,
    user_test_user.shape,
    prod_tfidf_user.shape]

    #predict ratings
    #test_predictions = model.predict([test.user_idx.astype(float).values, test.prod_idx.astype(float).values])
    test_predictions = model.predict([user_test_user.astype(float).values, 
                                      prod_test_user.astype(float).values,
                                      prod_tfidf_user.astype(float).values])
    
    test['p_ratings'] = test_predictions
    
    #print prediction distribution
    if(verbose):
        plt.figure()
        test.p_ratings.round(2).hist()
        
    #top predicted products by rating
    df_columns = ['asin', 'description', 'title', 'category2_t', 'category3_t',
       'category4_t', 'category5_t', 'category6_t', 'hasDescription',
       'price_t', 'containsAnySalesRank', 'numberQuestions', 'numberReviews',
       'meanStarRating', 'Category_', 'cat_idx', 'idx']
    y = pd.merge(products_clean.loc[:,df_columns], test.sort_values(
            by='p_ratings', ascending=False).head(10)[['prodIdx','p_ratings']], 
                 how='inner', left_on='idx', right_on='prodIdx')
    y.drop(['idx'], axis=1, inplace=True)
    
    #products reviewed by reviewer
    x = pd.merge(products_clean.loc[:,df_columns], 
             merged_3.loc[merged_3.reviewerID == rev_id,['reviewerID','asin',
                                                         'userIdx','prodIdx']], 
             how='inner', on='asin')
                 
    return x, y

print('Script: 08.01.01 [Create sim function] completed')
# =============================================================================
# 08.02.01 | Create predictions for unreviewed products by specified reviewer
# =============================================================================
#control variables
reviewers = 2
numPredictions = 10
pd.set_option('display.max_columns', 50)

#get random reviewer ID
def get_predictions_by_random_reviewer():
    rev_id = merged_3.reviewerID.value_counts().index[random.randint(0,merged_3.reviewerID.nunique())]
    x, y = get_sim_products_by_user(rev_id)
    print("------------------------------------------------")
    print("Products reviewed by {}:".format(rev_id))
    print(x[['prodIdx','asin','description',
             'title','category2_t', 'category3_t',
             'price_t','meanStarRating']])
    print("------------------------------------------------")
    print("Top {} products for {}:".format(numPredictions, rev_id))
    print(y[['prodIdx','asin','description',
             'title','category2_t', 'category3_t',
             'price_t','meanStarRating',
             'p_ratings']].sort_values(by='p_ratings', 
    ascending=False).head(numPredictions))
    print("------------------------------------------------\n")
    
#loop through reviewer predictions
for x in range(2):
    print("Get Reviews - {} of {}".format(x+1, reviewers))
    get_predictions_by_random_reviewer()
    
print('Script: 08.02.01 [Create predictions for reviewers] completed')
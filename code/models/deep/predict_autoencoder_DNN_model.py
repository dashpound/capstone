# ===============================================================================
# 10.00.01 | Predict ratings with Deep Neural Network Content Filtering Model | Documentation
# ===============================================================================
# Name:               10_Predict_Collaborative_DNN_model
# Author:             Merrill
# Last Edited Date:   11/17/19
# Description:        Build DNN model using only product data
#                     
# Notes:              Built model with both IDs only and full metadata; full 
#                     metadata model overfit data returning same similarity for
#                     all users
#
# Warnings:
#
# Outline:            Imports needed packages and set seed for reproducibility.
#                     Import dataframes for predictions
#                     Predict user ratings
#
#
# =============================================================================
# 10.00.02 | Import packages
# =============================================================================
# Import packages
#import pickle, datetime, os, gc
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from compress_pickle import load
#from pathlib import Path

# Import modules (other scripts)
from code.configuration.environment_configuration import *
#from clean_data_load import *
#from data_load import *

#print output
verbose = False
create_data_files = True

#display TSNE visuals - very time intenstive
show_visuals = True

#which dist function to use (sklearn or scpiy)
use_sklearn = True

#load encoded items and products
encoded_items = np.load('./data/pickles/enhanced/dnn_encoded_items_20191116.npy', allow_pickle=True)
products_clean = pd.read_pickle("./data/pickles/enhanced/dnn_products_clean_idx_all.pkl")
#prod_df = pd.read_pickle("./data/pickles/enhanced/dnn_prod_df.pkl")
prod_df = load("./data/pickles/enhanced/dnn_prod_df.gz")

print('Script: 08.00.02 [Import packages] completed')

# =============================================================================
# 10.02.01 | Create Scipy cosine and euclidean distance function
# =============================================================================
import scipy.spatial.distance
from sklearn.preprocessing import MinMaxScaler
def scipy_dist(matrix, vector, metric='euclidean'):
    """
    Compute the cosine distances between each row of matrix and vector.
    """
    v = vector.reshape(1, -1)
    d = scipy.spatial.distance.cdist(matrix, v, metric).reshape(-1)
    if metric == 'euclidean':
        return MinMaxScaler().fit_transform((d*-1).reshape(-1,1))
    else: 
        return 1-d

print('Script: 10.02.01 [Create scipy dist. func] completed')

# =============================================================================
# 10.03.01 | Setup sklearn cosine distance function
# =============================================================================

from sklearn.metrics.pairwise import cosine_similarity

print('Script: 10.03.01 [Imported sklearn cosine dist. func] completed')

# =============================================================================
# 10.04.01 | Find similar products by a given product
# =============================================================================
if(create_data_files):
    dash_predict = pd.DataFrame()

iterations = 20
predictions = 11
for j in range(iterations):
    i = random.randint(0,len(encoded_items))
    x = encoded_items[i]
    pd.set_option('display.max_columns', 50)
    sim_df = None
    
    if(use_sklearn):
        sim_sk = cosine_similarity(encoded_items, encoded_items[i].reshape(1,-1))
        sim_sk_results = pd.DataFrame(sim_sk, columns=['cdist'])
        sim_sk_df = pd.merge(products_clean, sim_sk_results.nlargest(predictions, 'cdist'), how='inner',
                          left_index=True, right_index=True)
        sim_df = sim_sk_df
    else:
        #products_clean.loc[products_clean.asin==prod_df.iloc[i].name,:]
        sim_cpy = scipy_dist(encoded_items, x)
        sim_cpy_results = pd.DataFrame(sim_cpy, columns=['cdist'])
        sim_cpy_df = pd.merge(products_clean, sim_cpy_results.nlargest(predictions, 'cdist'), how='inner',
                          left_index=True, right_index=True)
        sim_df = sim_cpy_df
    
    cols = ['cdist', 'asin', 'description', 'title', 'category2_t', 'category3_t',
               #'category4_t', 'category5_t', 'category6_t', 
               'price_t', 'numberQuestions', 'numberReviews',
               'meanStarRating', 'Category_', 'cat_idx', 'idx']
    
    if(create_data_files):
        orig_product = pd.DataFrame([products_clean.loc[i,['idx','asin']]]*predictions).reset_index(drop=True)
        pred_product = sim_df[['idx', 'asin', 'cdist'
        ]].sort_values(by = 'cdist', ascending = False).reset_index(drop=True)
        dash_predict = dash_predict.append(pd.merge(orig_product, pred_product, 
                                                    how='inner', left_index=True, 
                                                    right_index=True))
    if(verbose):
        print("--------------------------------------------------")
        print(sim_df[cols].sort_values(by='cdist', ascending=False))

if(create_data_files):
    dash_predict.rename(columns={"idx_x": "original_product_id_int",
                                 "asin_x": "original_product_id",
                                 "idx_y": "recommended_product_id_int",
                                 "asin_y": "recommended_product_id",
                                 "cdist": "predicted_rating"}, inplace=True)
    dash_predict.reset_index(drop=True, inplace=True)
    dash_predict = dash_predict[dash_predict.original_product_id_int!=dash_predict.recommended_product_id_int]
    dash_predict.to_pickle('./data/pickles/enhanced/dnn_autoencoder_20_predictions.pkl')
print('Script: 10.04.01 [Find similar products] completed')
# =============================================================================
# 10.05.01 | Visualize similar items using TSNE
# =============================================================================
if(show_visuals):
    #load encoded items numpy pickle
    item_tsne = np.load('./data/pickles/enhanced/dnn_encoded_item_tsne_20191116.npy', allow_pickle=True)
    
    #create categories list
    cats_df = products_clean['Category_'].value_counts()
    
    hsv = plt.get_cmap('hsv')
    colors = hsv(np.linspace(0, 3, len(cats_df)))
    color_cats = products_clean.cat_idx-1
    plt.figure(figsize=(10, 10))
    plt.scatter(item_tsne[:, 0], item_tsne[:, 1],
               c=colors[color_cats])
    plt.scatter(item_tsne[sim_df.index[0], 0], item_tsne[sim_df.index[0], 1],
           c='black', marker="*")
    plt.scatter(item_tsne[sim_df.index[1:], 0], item_tsne[sim_df.index[1:], 1],
           c='black', marker="P")
    plt.xticks(()); plt.yticks(())
    plt.show()    
    print('Script: 10.05.01 [Visualize embedding using TSNE] completed')              
else:
    print('Script: 10.05.01 [Visualize embedding using TSNE] skipped')    
print('Script: 10.05.01 [Visualize similar items] completed')
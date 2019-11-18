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
# 10.02.01 | Create Scipy cosine distance function
# =============================================================================
import scipy.spatial.distance
def cos_cdist(matrix, vector):
    """
    Compute the cosine distances between each row of matrix and vector.
    """
    v = vector.reshape(1, -1)
    return 1-scipy.spatial.distance.cdist(matrix, v, 'cosine').reshape(-1)

print('Script: 10.02.01 [Create scipy cosine dist. func] completed')

# =============================================================================
# 10.03.01 | Setup sklearn cosine distance function
# =============================================================================

from sklearn.metrics.pairwise import cosine_similarity

print('Script: 10.03.01 [Imported sklearn cosine dist. func] completed')

# =============================================================================
# 10.04.01 | Find similar products by a given product
# =============================================================================
i = random.randint(0,len(encoded_items))
x = encoded_items[i]
pd.set_option('display.max_columns', 50)
sim_df = None

if(use_sklearn):
    sim_sk = cosine_similarity(encoded_items, encoded_items[i].reshape(1,-1))
    sim_sk_results = pd.DataFrame(sim_sk, columns=['cdist'])
    sim_sk_df = pd.merge(products_clean, sim_sk_results.nlargest(10, 'cdist'), how='inner',
                      left_index=True, right_index=True)
    sim_df = sim_sk_df
else:
    #products_clean.loc[products_clean.asin==prod_df.iloc[i].name,:]
    sim_cpy = cos_cdist(encoded_items, x)
    sim_cpy_results = pd.DataFrame(sim_cpy, columns=['cdist'])
    sim_cpy_df = pd.merge(products_clean, sim_cpy_results.nlargest(10, 'cdist'), how='inner',
                      left_index=True, right_index=True)
    sim_df = sim_cpy_df

cols = ['cdist', 'asin', 'description', 'title', 'category2_t', 'category3_t',
           'category4_t', 'category5_t', 'category6_t', 
           'price_t', 'numberQuestions', 'numberReviews',
           'meanStarRating', 'Category_', 'cat_idx', 'idx']
print(sim_df[cols].sort_values(by='cdist', ascending=False))
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
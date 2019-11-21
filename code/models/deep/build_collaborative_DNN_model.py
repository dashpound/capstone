# ===============================================================================
# 07.00.01 | Deep Neural Network Collaborative Filtering Model | Documentation
# ===============================================================================
# Name:               07_Build_Collaborative_DNN_model
# Author:             Merrill
# Last Edited Date:   11/9/19
# Description:        Build DNN model using only user ratings and product data
#                     
# Notes:              Built model with both IDs only and full metadata; full 
# Notes:              Built model with both IDs only, full, and partial metadata; full 
#                     metadata model overfit data returning same similarity for
#                     all users
#
# Warnings:
#
#
# Outline:            Imports needed packages and set seed for reproducibility.
#                     Prepare data for modeling by filtering and merging user/product data
#                     Create DNN model with embeddings
#                     Train DNN model with embeddings
#                     Save model, embeddings, and updated products DF
#
#
# =============================================================================
# 07.00.02 | Import packages
# =============================================================================
# Import packages
import pickle, datetime, os, gc
from compress_pickle import dump
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# Import modules (other scripts)
from code.configuration.environment_configuration import *
from code.dataprep.clean_data_load import *
from code.dataprep.data_load import *

#console output flag
verbose = True
#create data files flag
create_data_files = False

print('Script: 07.00.02 [Import packages] completed')
# =============================================================================
# 07.01.01 | Prepare data for modeling
# =============================================================================

#add category grouping ID
cats = products_clean.category2_t+"-"+products_clean.category3_t
products_clean.loc[:,'Category_'] = cats.copy()
cats_df = pd.DataFrame(cats.value_counts(), columns=['Count'])
cats_df.reset_index(inplace=True)
cats_df['cat_idx'] = cats_df.index + 1
cats_df.rename(columns={"index": "Category_"}, inplace=True)
products_clean = pd.merge(products_clean, cats_df[['Category_','cat_idx']], 
                   how='inner', on='Category_').copy()

#quantitative columns - commented out columns causing overfitting
products_prep = products_clean[['asin', #'hasDescription', 
                                'price_t', #'containsAnySalesRank',
                                'numberQuestions','numberReviews',
                                'meanStarRating', 'cat_idx'#,'star1Rating','star2Rating',
                                #'star3Rating','star4Rating','star5Rating'
                                ]].copy()

#quantitative columns - commented out columns causing overfitting
reviews_prep = reviews_df[['reviewerID','asin','helpful_proportion','helpful_numer','helpful_denom',
                           'overall','reviewYear','reviewMonth']].copy()

reviews_prep.loc[:,'help_count']=reviews_prep.loc[:,
                ['helpful_numer','helpful_denom']].sum(axis=1).copy()
reviews_prep.drop(columns=['helpful_numer','helpful_denom'], inplace=True)

## deprecated - pickle changed 11.10.2019
#reviews_prep = reviews_df[['reviewerID','asin','helpful',
#                           'overall','reviewTime']]
#rt = reviews_prep.reviewTime.str.split(pat="[ ,]").copy()
#reviews_prep.loc[:, 'reviewYear'] = rt.apply(lambda x: x[1]).copy()
#reviews_prep.loc[:, 'reviewMonth'] = rt.apply(lambda x: x[0]).copy()
#reviews_prep.drop(['reviewTime'], axis=1, inplace=True)

#create help metric variables

## deprecated based on updated reviews_df
#reviews_prep.loc[:,'help_ratio']=reviews_prep['helpful'].apply(lambda x: (x[0]+1)/(x[1]+1))
#reviews_prep.loc[:,'help_count']=reviews_prep['helpful'].apply(lambda x: x[0]+x[1])

#zero out ratios with no counts due to +1 above
#reviews_prep.loc[reviews_prep.help_count==0,'help_ratio'] = 0.0

#create merged review / products dataframe
merged = pd.merge(reviews_prep, products_prep, how='inner', on='asin').copy()
#merged.drop(['helpful'], axis=1, inplace=True)

with open(Path(working_directory + data_path + '/pickles/enhanced/reviews_meta_combined_aggregated.pkl'), 'rb') as pickle_file:
    rev_meta_agg_df = pickle.load(pickle_file)
    rev_meta_agg_df = pd.DataFrame(rev_meta_agg_df)

#read tf-idf matrix
prod_tfidf_df = pd.read_csv('./data/tfidf/electronics/product/electronics_product_tfidf.csv')
prod_tfidf_df.rename(columns={"Unnamed: 0": "asin"}, inplace=True)
prod_tfidf_df = pd.merge(prod_tfidf_df, products_clean.asin, how='inner', on='asin')
#merge in reviewer metadata
merged_2 = pd.merge(merged, rev_meta_agg_df, how='inner', on='reviewerID').copy()
del rev_meta_agg_df
## unneeded due to explicit column name filtering above
#drop one hot encoded columns - they don't work well with DNNs
#merged_2 = merged_2[merged_2.columns[~merged_2.columns.isin(['hasDescription','containsAnySalesRank'])]]

#create unique ID indexes
u_uniq = merged_2.reviewerID.unique()
user2idx = {o:i for i,o in enumerate(u_uniq)}
merged_2['userIdx'] = merged_2.reviewerID.apply(lambda x: user2idx[x]).copy()

p_uniq = merged_2.asin.unique()
prod2idx = {o:i for i,o in enumerate(p_uniq)}
merged_2['prodIdx'] = merged_2.asin.apply(lambda x: prod2idx[x]).copy()

merged_3 = pd.merge(merged_2, prod_tfidf_df, how='inner', on='asin').copy()

#create index values for users and products
user2idx_df = pd.DataFrame.from_dict(user2idx, orient='index', columns=['idx'])
prod2idx_df = pd.DataFrame.from_dict(prod2idx, orient='index', columns=['idx'])

#merge in index on products_clean
products_clean['idx'] = products_clean.asin.apply(lambda x: prod2idx_df.loc[x])

if(create_data_files):
    #pickle indexes
    user2idx_df.to_pickle("./data/pickles/enhanced/user2idx_20191114.pkl")
    prod2idx_df.to_pickle("./data/pickles/enhanced/prod2idx_20191114.pkl") 
    products_clean.to_pickle("./data/pickles/enhanced/dnn_products_clean_idx_all.pkl")

if(verbose):
    #check number of users and products
    print("Users: {}, Products: {}".format(len(u_uniq), len(p_uniq)))


#scale data
scaler = MinMaxScaler()
columns_to_scale = ['helpful_proportion', 'help_count', 'reviewYear',
       'reviewMonth', 'price_t', 'numberQuestions', 'numberReviews',
       'meanStarRating', 'MaxRating', 'MinRating', 'NumberOfRatings',
       'AverageRating', 'MedianRating', 'SummedRatings', 'MaxPrice',
       'MinPrice', 'AveragePrice', 'MedianPrice', 'SummedPrice',
       'SummedHelpfulNumer', 'SummedHelpfulDenom', 'MaxNumDaysBetweenReviews',
       'MinNumDaysBetweenReviews', 'AverageNumDaysBetweenReviews',
       'MedianNumDaysBetweenReviews', 'SummedNumDaysBetweenReviews',
       'cat_idx']
merged_2.loc[:,columns_to_scale] = scaler.fit_transform(merged_2.loc[:,columns_to_scale]).copy()

#products frame
prod_df = merged_2.loc[:,['prodIdx', 'price_t',
                           'numberQuestions', 'numberReviews', 
                           'meanStarRating','cat_idx']].copy()
tfidf_df = merged_3.iloc[:,30:].copy()

#user frame
user_df = merged_2.loc[:,['userIdx', 'helpful_proportion', 'help_count', 'reviewYear', 'reviewMonth', 
                          'MaxRating', 'MinRating', 'NumberOfRatings', 'AverageRating', 
                          #'MedianRating', 'SummedRatings', 
                          'MaxPrice', 'MinPrice', 'AveragePrice'#, 'MedianPrice', 'SummedPrice',
                          #'SummedHelpfulNumer', 'SummedHelpfulDenom', 
                          #'MaxNumDaysBetweenReviews', 'MinNumDaysBetweenReviews', 
                          #'AverageNumDaysBetweenReviews'#
                          #'MedianNumDaysBetweenReviews', 'SummedNumDaysBetweenReviews'
                          ]].copy()

#ratings frame
ratings_df = merged_2['overall'].copy()

if(create_data_files):
    
    #write to gunzip pickle 
    dump(merged_3,"./data/pickles/enhanced/dnn_merged_3.gz")
    dump(user_df,"./data/pickles/enhanced/dnn_user_df.gz")
    dump(prod_df,"./data/pickles/enhanced/dnn_prod_df.gz")
    dump(tfidf_df,"./data/pickles/enhanced/dnn_tfidf_df.gz")
    """
    merged_3.to_pickle("./data/pickles/enhanced/dnn_merged_3.pkl")
    user_df.to_pickle("./data/pickles/enhanced/dnn_user_df.pkl")
    prod_df.to_pickle("./data/pickles/enhanced/dnn_prod_df.pkl")
    tfidf_df.to_pickle("./data/pickles/enhanced/dnn_tfidf_df.pkl")
    """

#drop prodIdx from tfidf as not needed for modeling but rather for future reference
tfidf_df.drop(columns=['userIdx','prodIdx'], inplace=True)
print('Script: 07.01.01 [Prepare data for modeling] completed')

# =============================================================================
# 07.02.01 | Create DNN model with user and product embeddings
# =============================================================================
#import keras libraries
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K

#clear prior session before running
K.clear_session()
model = None
user_cols = user_df.shape[1]
prod_cols = prod_df.shape[1]
tfidf_cols = tfidf_df.shape[1]

item_cols = prod_cols + tfidf_cols
#user_cols = 1
#prod_cols = 1
user_dim = user_df.shape[0]
user_input = layers.Input(shape=(user_cols,), name = 'user')
item_input = layers.Input(shape=(item_cols,), name = 'item')
#tfidf_input = layers.Input(shape=(tfidf_cols,), name = 'tfidf')

#starting at 100
def get_embedding_size(m):
    return(int(min(50, round((m+1)/2,0))))
#embedding_size = 50

user_dim = get_embedding_size(user_cols)
prod_dim = get_embedding_size(prod_cols)
tfidf_dim = get_embedding_size(tfidf_cols)

item_dim = get_embedding_size(prod_cols+tfidf_cols)
# input_dim: int > 0. Size of the vocabulary, i.e. maximum integer index + 1.
# output_dim: int >= 0. Dimension of the dense embedding.
# input_length: Length of input sequences, when it is constant. 
user_embedding = layers.Embedding(output_dim = user_dim, input_dim = len(u_uniq),
                           input_length = user_cols, name = 'user_embedding')(user_input)
item_embedding = layers.Embedding(output_dim = item_dim, input_dim = len(p_uniq),
                           input_length = item_cols, name = 'item_embedding')(item_input)
#tfidf_embedding = layers.Embedding(output_dim = tfidf_dim, input_dim = len(p_uniq)+1,
#                           input_length = tfidf_cols, name = 'tfidf_embedding')(tfidf_input)

# reshape from shape: (samples, input_length, embedding_size)
# to shape: (samples, input_length * embedding_size) which is
# equal to shape: (samples, embedding_size)
user_vecs = layers.Flatten(name="FlattenUser")(user_embedding)
user_vecs_d = layers.Dropout(0.25, name="Dropout_u_1")(user_vecs)
item_vecs = layers.Flatten(name="FlattenItem")(item_embedding)
item_vecs_d = layers.Dropout(0.25, name="Dropout_i_1")(item_vecs)
#tfidf_vecs = layers.Flatten(name="FlattenTfidf")(tfidf_embedding)
#tfidf_vecs_d = layers.Dropout(0.25, name="Dropout_t_1")(tfidf_vecs)

# concatenate user_vecs and item_vecs
input_vecs = layers.Concatenate(name="Concat")([user_vecs, item_vecs])#,tfidf_vecs])
input_vecs = layers.Dropout(0.5, name="Dropout_2")(input_vecs)

# Include RELU as activation layer
x = layers.Dense(72, activation='relu',name="Dense_1")(input_vecs)
x = layers.Dropout(0.4, name="Dropout_3")(x)
x = layers.Dense(36, activation='relu',name="Dense_2")(x)
x = layers.Dropout(0.25, name="Dropout_4")(x)
#x = layers.Dense(32, activation='relu',name="Dense_3")(x)
#x = layers.Dropout(0.25, name="Dropout_5")(x)
#x = layers.Dense(10, activation='relu',name="Dense_4")(x)
#x = layers.Dropout(0.5, name="Dropout_6")(x)
y = layers.Dense(1, activation='sigmoid', name="output")(x) * 5 + 0.5 #scale to 1-5

#custom r2 metric
def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()))

#custom rmse metric
def rmse(y_true, y_pred):
	return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

model = keras.models.Model(inputs=[user_input, item_input],
    #                               tfidf_input], 
    outputs=y)
model.compile(optimizer='adam', loss='mae', metrics=['mse','mape'])


#add tensorboard
#log_dir="logs/fit/recsys_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join(
    "logs",
    "fit",
    "recsys_tfidf_combined_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
)
if(verbose):
    print("Tensorboard callbacks directory:",log_dir)

tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)

print('Script: 07.02.01 [Create DNN model w/ embeddings] completed')
# =============================================================================
# 07.03.01 | Train DNN model with user and product embeddings
# =============================================================================
# input user_id_train & item_id_train
# output rating_train

item_df = pd.merge(prod_df, tfidf_df, how='inner', left_index=True, right_index=True)
history = model.fit([user_df.astype(float).values, item_df.astype(float).values],
                     #tfidf_df.astype(float).values], 
                     ratings_df.astype(float).values,
                     batch_size=64, epochs=10, validation_split=0.1,
                     shuffle=True,callbacks=[tensorboard_callback])

# get model weights and shape
weights = model.get_weights()
user_embeddings = weights[0]
item_embeddings = weights[1]
#tfidf_embeddings = weights[2]

if(verbose):
    print("layer shapes:")
    print([w.shape for w in weights])
    
    # Model Summary
    print(model.summary())


"""
#testing other layer encodings
#get dense encoded weights
encoder = keras.models.Model(inputs=[user_input, item_input, tfidf_input], outputs=input_vecs)
encoder.summary()

import numpy as np
encoded_items = None
for j in range(0, len(user_df)-1,50000):
    if encoded_items is None:
        encoded_items = encoder.predict([user_df.values[:50000], 
                                 prod_df.values[:50000],
                                 tfidf_df.values[:50000]])
    else:
        encoded_items = np.append(encoded_items, encoder.predict([user_df.values[j:j+50000], 
                                 prod_df.values[j:j+50000],
                                 tfidf_df.values[j:j+50000]]), axis=0)
"""   
print('Script: 07.03.01 [Train DNN model w/ embeddings] completed')
# =============================================================================
# 07.04.01 | Save DNN model, user and product embeddings
# =============================================================================
if(create_data_files):
    #save model weights
    model.save('./code/models/deep/models/merged_2embedding_2L_min_func_f_20191120_cat_nlp.h5')
    
    #save embeddings
    pd.DataFrame(user_embeddings).to_pickle("./data/pickles/enhanced/user_embeddings_2L_minf_20191120_cat_nlp.pkl")
    pd.DataFrame(item_embeddings).to_pickle("./data/pickles/enhanced/product_embeddings_2L_minf_20191120_cat_nlp.pkl")
    #pd.DataFrame(tfidf_embeddings).to_pickle("./data/pickles/enhanced/product_tfidf_embeddings_2L_64f_20191120_cat_nlp.pkl")
    
else:
    print("data file creation skipped..")

print('Script: 07.04.01 [Save DNN model and embeddings] completed')
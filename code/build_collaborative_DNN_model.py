# ===============================================================================
# 07.00.01 | Deep Neural Network Collaborative Filtering Model | Documentation
# ===============================================================================
# Name:               07_Build_Collaborative_DNN_model
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
# 07.00.02 | Import packages
# =============================================================================
# Import packages
import pickle, datetime, os, gc
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# Import modules (other scripts)
from environment_configuration import *
from clean_data_load import *
from data_load import *

#console output flag
verbose = False
#create data files flag
create_data_files = False

print('Script: 07.00.02 [Import packages] completed')
# =============================================================================
# 07.01.01 | Prepare data for modeling
# =============================================================================
#quantitative columns - commented out columns causing overfitting
products_prep = products_clean[['asin', #'hasDescription', 
                                'price_t', #'containsAnySalesRank',
                                'numberQuestions','numberReviews',
                                'meanStarRating'#,'star1Rating','star2Rating',
                                #'star3Rating','star4Rating','star5Rating'
                                ]]

#quantitative columns - commented out columns causing overfitting
reviews_prep = reviews_df[['reviewerID','asin','helpful','overall','unixReviewTime']]
#create help metric variables

reviews_prep.loc[:,'help_ratio']=reviews_prep['helpful'].apply(lambda x: (x[0]+1)/(x[1]+1))
reviews_prep.loc[:,'help_count']=reviews_prep['helpful'].apply(lambda x: x[0]+x[1])

#zero out ratios with no counts due to +1 above
reviews_prep.loc[reviews_prep.help_count==0,'help_ratio'] = 0.0

#create merged review / products dataframe
merged = pd.merge(reviews_prep, products_prep, how='inner', on='asin')
merged.drop(['helpful'], axis=1, inplace=True)

with open(Path(working_directory + data_path + '/reviews_meta_combined_aggregated.pkl'), 'rb') as pickle_file:
    rev_meta_agg_df = pickle.load(pickle_file)
    rev_meta_agg_df = pd.DataFrame(rev_meta_agg_df)

#merge in reviewer metadata
merged_2 = pd.merge(merged, rev_meta_agg_df, how='inner', on='reviewerID')

#drop one hot encoded columns - they don't work well with DNNs
merged_2 = merged_2[merged_2.columns[~merged_2.columns.isin(['hasDescription','containsAnySalesRank'])]]

#create unique ID indexes
u_uniq = merged_2.reviewerID.unique()
user2idx = {o:i for i,o in enumerate(u_uniq)}
merged_2['userIdx'] = merged_2.reviewerID.apply(lambda x: user2idx[x])

p_uniq = merged_2.asin.unique()
prod2idx = {o:i for i,o in enumerate(p_uniq)}
merged_2['prodIdx'] = merged_2.asin.apply(lambda x: prod2idx[x])

#create index values for users and products
user2idx_df = pd.DataFrame.from_dict(user2idx, orient='index', columns=['idx'])
prod2idx_df = pd.DataFrame.from_dict(prod2idx, orient='index', columns=['idx'])


if(create_data_files):
    #pickle indexes
    user2idx_df.to_pickle("../data/user2idx_20191106.pkl")
    prod2idx_df.to_pickle("../data/prod2idx_20191106.pkl")

if(verbose):
    #check number of users and products
    print(len(u_uniq), len(p_uniq))

#scale data
scaler = MinMaxScaler()
merged_2.iloc[:,3:28] = scaler.fit_transform(merged_2.iloc[:,3:28])

#products frame
prod_df = merged_2.loc[:,['prodIdx', 'unixReviewTime', 'price_t',
                           'numberQuestions', 'numberReviews', 
                           'meanStarRating']]


#user frame
user_df = merged_2.loc[:,['userIdx', 'MaxRating', 'MinRating', 'NumberOfRatings', 
                           'AverageRating', 'MedianRating', 'SummedRatings', 'MaxPrice', 
                           'MinPrice', 'AveragePrice', #'MedianPrice', 'SummedPrice', 
                           #'SummedHelpfulNumer', 'SummedHelpfulDenom', 
                           'help_ratio', 'help_count', 
                           'MaxNumDaysBetweenReviews', 'MinNumDaysBetweenReviews', 
                           'AverageNumDaysBetweenReviews', 'MedianNumDaysBetweenReviews', 
                           'SummedNumDaysBetweenReviews']]

#ratings frame
ratings_df = merged_2['overall']

if(create_data_files):
    merged_2.to_pickle("../data/dnn_merged_2.pkl")
    user_df.to_pickle("../data/dnn_user_df.pkl")
    prod_df.to_pickle("../data/dnn_prod_df.pkl")

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

user_cols = user_df.shape[1]
prod_cols = prod_df.shape[1]
#user_cols = 1
#prod_cols = 1
user_dim = user_df.shape[0]
user_input = layers.Input(shape=(user_cols,), name = 'user')
item_input = layers.Input(shape=(prod_cols,), name = 'item')

#starting at 100
embedding_size = 50

# input_dim: int > 0. Size of the vocabulary, i.e. maximum integer index + 1.
# output_dim: int >= 0. Dimension of the dense embedding.
# input_length: Length of input sequences, when it is constant. 
user_embedding = layers.Embedding(output_dim = embedding_size, input_dim = len(u_uniq),
                           input_length = user_cols, name = 'user_embedding')(user_input)
item_embedding = layers.Embedding(output_dim = embedding_size, input_dim = len(p_uniq),
                           input_length = prod_cols, name = 'item_embedding')(item_input)

# reshape from shape: (samples, input_length, embedding_size)
# to shape: (samples, input_length * embedding_size) which is
# equal to shape: (samples, embedding_size)
user_vecs = layers.Flatten(name="FlattenUser")(user_embedding)
user_vecs_d = layers.Dropout(0.5, name="Dropout_u_1")(user_vecs)
item_vecs = layers.Flatten(name="FlattenItem")(item_embedding)
item_vecs_d = layers.Dropout(0.5, name="Dropout_i_1")(item_vecs)

# concatenate user_vecs and item_vecs
input_vecs = layers.Concatenate(name="Concat")([user_vecs_d, item_vecs_d])
input_vecs = layers.Dropout(0.5, name="Dropout_2")(input_vecs)

# Include RELU as activation layer
x = layers.Dense(64, activation='relu',name="Dense_1")(input_vecs)
x_d = layers.Dropout(0.5, name="Dropout_3")(x)
x_2 = layers.Dense(32, activation='relu',name="Dense_2")(x_d)
x_d2 = layers.Dropout(0.5, name="Dropout_4")(x_2)
y = layers.Dense(1, activation='sigmoid', name="output")(x_d2) * 5 + 0.5 #scale to 1-5

#custom r2 metric
def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

model = keras.models.Model(inputs=[user_input, item_input], outputs=y)
model.compile(optimizer='adam', loss='mae', metrics=[r2_keras])

#add tensorboard
#log_dir="logs/fit/recsys_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join(
    "logs",
    "fit",
    "recsys_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
)
if(verbose):
    print("Tensorboard callbacks directory:",log_dir)

tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

print('Script: 07.02.01 [Create DNN model w/ embeddings] completed')
# =============================================================================
# 07.03.01 | Train DNN model with user and product embeddings
# =============================================================================
# input user_id_train & item_id_train
# output rating_train
history = model.fit([user_df.astype(float).values, prod_df.astype(float).values], 
                     ratings_df.astype(float).values,
                     batch_size=256, epochs=30, validation_split=0.1,
                     shuffle=True,callbacks=[tensorboard_callback])

# get model weights and shape
weights = model.get_weights()
user_embeddings = weights[0]
item_embeddings = weights[1]

if(verbose):
    print("layer shapes:")
    [w.shape for w in weights]

    # Model Summary
    model.summary()

print('Script: 07.03.01 [Train DNN model w/ embeddings] completed')
# =============================================================================
# 07.04.01 | Save DNN model, user and product embeddings
# =============================================================================
if(create_data_files):
    #save model weights
    model.save('merged_2_embedding_all_features_50f.h5')
    
    #save embeddings
    pd.DataFrame(user_embeddings).to_pickle("../data/user_embeddings_all_features_50f.pkl")
    pd.DataFrame(item_embeddings).to_pickle("../data/product_embeddings_all_features_50f.pkl")
    
    #merge in index on products_clean
    products_clean['idx'] = products_clean.asin.apply(lambda x: prod2idx_df.loc[x])
    products_clean.to_pickle("../data/dnn_products_clean_idx_all.pkl")
else:
    print("data file creation skipped..")

print('Script: 07.04.01 [Save DNN model and embeddings] completed')    
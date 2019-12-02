# ===============================================================================
# 09.00.01 | Deep Neural Network Collaborative Filtering Model | Documentation
# ===============================================================================
# Name:               09_Build_Collaborative_DNN_model
# Author:             Merrill
# Last Edited Date:   11/15/19
# Description:        Build DNN model using product data
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
# 09.00.02 | Import packages
# =============================================================================
# Import packages
import datetime, os
import numpy as np
import pandas as pd
from compress_pickle import load

# Import modules (other scripts)
from code.configuration.environment_configuration import *
#from code.dataprep.clean_data_load import *
#from code.dataprep.data_load import *

#console output flag
verbose = True
#create data files flag
create_data_files = True
#display TSNE visuals - very time intenstive
show_visuals = False

print('Script: 09.00.02 [Import packages] completed')
# =============================================================================
# 09.01.01 | Prepare data for modeling
# =============================================================================
#load pickles
products_clean = pd.read_pickle("./data/pickles/enhanced/dnn_products_clean_idx_all.pkl")
#prod_df = pd.read_pickle("./data/pickles/enhanced/dnn_prod_df.pkl")
#tfidf_df = pd.read_pickle("./data/pickles/enhanced/dnn_tfidf_df.pkl")
#merged_3 = pd.read_pickle("./data/pickles/enhanced/dnn_merged_3.pkl")
merged_3 = load("./data/pickles/enhanced/dnn_merged_3.gz")

#select product columns and tfidf from merged_3
#prod_df = pd.merge(merged_3[['asin', 'prodIdx','price_t', 'numberQuestions','numberReviews',
#                          'meanStarRating','cat_idx']],
#                   merged_3.iloc[:,31:], how='inner', on='prodIdx')#.drop_duplicates()


prod_tfidf_df = pd.read_csv('./data/tfidf/electronics/product/electronics_product_tfidf.csv')
prod_tfidf_df.rename(columns={"Unnamed: 0": "asin"}, inplace=True)

prod_df = pd.merge(products_clean[['asin', 'idx','price_t', 'numberQuestions','numberReviews',
                          'meanStarRating','cat_idx']], prod_tfidf_df, how='inner', on='asin')
prod_df.set_index('asin', inplace=True)
prod_df.drop(columns=['idx'], inplace=True)

print("Product dataframe shape: {}".format(prod_df.shape))
print('Script: 09.01.01 [Prepare data for modeling] completed')

# =============================================================================
# 09.02.01 | Create autoencoder DNN model product embeddings
# =============================================================================
#import packages
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K

# input_dim: int > 0. Size of the vocabulary, i.e. maximum integer index + 1.
# output_dim: int >= 0. Dimension of the dense embedding.
# input_length: Length of input sequences, when it is constant. 

#clear prior session before running
K.clear_session()

num_prods, prod_cols = prod_df.shape
item_input = layers.Input(shape=(prod_cols,), name = 'item')
#"""
# original dense layer approach for autoencoder
# Include RELU as activation layer
#encode = layers.Dense(125, activation='relu',name="Dense_1e")(item_input)
#encode = layers.Dropout(0.25, name="Dropout_1e")(encode)
encode = layers.Dense(100, activation='relu',name="Dense_2e")(item_input)
#encode = layers.Dropout(0.1, name="Dropout_2e")(encode)
encode = layers.Dense(75, activation='relu',name="Dense_3e")(encode)
#encode = layers.Dropout(0.05, name="Dropout_3e")(encode)
#"""

#starting at 100
#embedding_size = 50
#item_embedding = layers.Embedding(output_dim = embedding_size, input_dim = num_prods,
#                           input_length = prod_cols, name = 'item_embedding')(item_input)

#"""
encode = layers.Dense(50, activation='relu',name="Dense_4e")(encode)
#"""

#decode = layers.Flatten(name="Flatten_User")(item_embedding)
# Include RELU as activation layer
decode = layers.Dense(75, activation='relu',name="Dense_1d")(encode)
#decode = layers.Dropout(0.05, name="Dropout_1d")(decode)
decode = layers.Dense(100, activation='relu',name="Dense_2d")(decode)
#decode = layers.Dropout(0.1, name="Dropout_2d")(decode)
#decode = layers.Dense(125, activation='relu',name="Dense_3d")(decode)
#decode = layers.Dropout(0.25, name="Dropout_3d")(decode)
y = layers.Dense(prod_cols, activation='relu', name="output")(decode)# * 5 + 0.5 #scale to 1-5

def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

model = keras.models.Model(inputs=item_input, outputs=y)
model.compile(optimizer='adam', loss='mae', metrics=[r2_keras])

#add tensorboard
#log_dir="logs/fit/recsys_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join(
    "logs",
    "fit",
    "recsys_autoencoder_dense_2L_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
)
print(log_dir)
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
print('Script: 09.02.01 [Build autoencoder model] completed')
# =============================================================================
# 09.03.01 | Train DNN model with user and product embeddings
# =============================================================================
# input user_id_train & item_id_train
# output rating_train
history = model.fit([prod_df.astype(float).values], prod_df.astype(float).values,
                    batch_size=64, epochs=10, validation_split=0.1,
                    shuffle=True,callbacks=[tensorboard_callback])

#get dense encoded weights
encoder = keras.models.Model(inputs=item_input, outputs=encode)
encoder.summary()
encoded_items = encoder.predict(prod_df.astype(float).values)

# get model weights and shape
weights = model.get_weights()

#get embeddings weights
item_embeddings = weights[0]

if(verbose):
    print("layer shapes:")
    [w.shape for w in weights]

    # Model Summary
    print(model.summary())

if(create_data_files):
    #dense model
    model.save('./code/models/deep/models/autoencoder_2L_dense_no_drop_product_50f_20191116_cat_nlp.h5')
    encoder.save('./code/models/deep/models/autoencoder_encoder_only_2L_dense_no_drop_product_50f_20191116_cat_nlp.h5')
    np.save('./data/pickles/enhanced/dnn_encoded_items_20191116.npy', arr=encoded_items, allow_pickle=True)

print('Script: 09.03.01 [Train dense autoencoder model] completed')    

# =============================================================================
# 09.04.01 | Visualize embedding using TSNE
# =============================================================================
if(show_visuals):
    from sklearn.manifold import TSNE
    #item_tsne = TSNE(perplexity=30).fit_transform(item_embeddings)
    item_tsne = TSNE(perplexity=30).fit_transform(encoded_items)
    
    np.save('./data/pickles/enhanced/dnn_encoded_item_tsne_20191116.npy', arr=item_tsne, allow_pickle=True)
    cats_df = products_clean['Category_'].value_counts()
    
    import matplotlib.pyplot as plt
    hsv = plt.get_cmap('hsv')
    colors = hsv(np.linspace(0, 5.0, len(cats_df)))
    color_cats = prod_df.cat_idx-1
    plt.figure(figsize=(10, 10))
    plt.scatter(item_tsne[:, 0], item_tsne[:, 1],
               c=colors[color_cats])
    plt.xticks(()); plt.yticks(())
    plt.show()    
    print('Script: 09.04.01 [Visualize embedding using TSNE] completed')              
else:
    print('Script: 09.04.01 [Visualize embedding using TSNE] skipped')    
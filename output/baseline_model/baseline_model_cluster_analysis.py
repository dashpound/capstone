# =============================================================================
# 10.00.02 | Import Packages
# =============================================================================
# Import packages
import pandas as pd
import pickle
import numpy as np
from pathlib import Path
import gc

# Import modules (other scripts)
from code.environment_configuration import working_directory, modeling_path, data_path, products_clean_path, reviews_path
from code.configuration.functions import conv_pivot2df

# =============================================================================
# 10.01.01 | Read in Data
# =============================================================================
# was having a hard time with pathing....
with open(Path(str(Path(working_directory).parents[0])+'/output'+'/baseline_model'+'/svd_recommendations_cluster0.pkl'), 'rb') as pickle_file:
    cluster0_df = pickle.load(pickle_file)
    cluster0_df = pd.DataFrame(cluster0_df)
    
with open(Path(str(Path(working_directory).parents[0])+'/output'+'/baseline_model'+'/svd_recommendations_cluster1.pkl'), 'rb') as pickle_file:
    cluster1_df = pickle.load(pickle_file)
    cluster1_df = pd.DataFrame(cluster1_df)
    
with open(Path(str(Path(working_directory).parents[0])+data_path+products_clean_path), 'rb') as pickle_file:
    product_df = pickle.load(pickle_file)
    product_df = pd.DataFrame(product_df) 
    
with open(Path(str(Path(working_directory).parents[0])+data_path+reviews_path), 'rb') as pickle_file:
    reviews_df = pickle.load(pickle_file)
    reviews_df = pd.DataFrame(reviews_df)
    
    
# =============================================================================
# 10.02.01 | Camera & Photos Data Refresher
# =============================================================================
reviews_sub = reviews_df[reviews_df['category2_t']=='Camera & Photo']

print('Unique Products: ',reviews_sub['asin'].nunique())
print('Unique Reviewers: ',reviews_sub['reviewerID'].nunique())
print('Rating Range: \n',reviews_sub['overall'].describe())

reviews_sub_sum = pd.pivot_table(reviews_sub, index=['reviewerID'], values=['overall'],
                            aggfunc={'overall': [len, np.min, np.mean, np.median, np.max]})
conv_pivot2df(reviews_sub_sum)
reviews_sub_sum['reviewerID'] = reviews_sub_sum.index

review_dist = reviews_sub_sum['len'].value_counts()

# =============================================================================
# 10.03.01 | HL Inspection
# =============================================================================
print('Unique Products: ',cluster0_df['asin'].nunique())
print('Unique Reviewers: ',cluster0_df['reviewerID'].nunique())
# seeing only 5's
print('Predicted Rating Range: \n',cluster0_df['predicted_rating'].describe())

print('Unique Products: ',cluster1_df['asin'].nunique())
print('Unique Reviewers: ',cluster1_df['reviewerID'].nunique())
# seeing only 5's
print('Predicted Rating Range: \n',cluster1_df['predicted_rating'].describe())


# =============================================================================
# 10.04.01 | Individual Recommendations | Cluster 0
# =============================================================================
cluster0_enhanced = pd.merge(cluster0_df, product_df, on='asin', how='inner')
cluster0_sub = cluster0_enhanced[['asin','reviewerID','predicted_rating','category2_t','category3_t','price_t']]
cluster0_sub = cluster0_sub.reset_index().sort_values(by=['reviewerID','predicted_rating'],ascending=False).reset_index(drop=True)
cluster0_sub = cluster0_sub.drop('index', axis=1)

# start picking predictions
# cluster0_sub['reviewerID'].head()
print('User 1:\n', cluster0_sub[cluster0_sub['reviewerID']=='A005721627VX5W2COKKK2'].to_string())

print('User 2:\n', cluster0_sub[cluster0_sub['reviewerID']=='A5HVB143T35CE'].to_string())

print('User 3:\n', cluster0_sub[cluster0_sub['reviewerID']=='A2GG7WQ5Q7NVAZ'].to_string())

print('User 4:\n', cluster0_sub[cluster0_sub['reviewerID']=='AKRCZLJG4OMZJ'].to_string())

print('User 5:\n', cluster0_sub[cluster0_sub['reviewerID']=='AWX8WC37VTMNX'].to_string())

# the recommendations are different for users but this makes sense b/c these are products they haven't seen before
cluster0_sum = cluster0_enhanced['category2_t'].value_counts()
cluster0_sum = cluster0_enhanced['reviewerID'].value_counts().to_frame('count') # just taking the top 10 recs for each user

# =============================================================================
# 10.04.02 | Cluster 0 Analysis
# =============================================================================
# let's look at user profiles  - based on what actually happened
cluster0_actual = pd.merge(cluster0_df['reviewerID'],reviews_df,on='reviewerID',how='inner')

cluster0_sum = pd.pivot_table(cluster0_actual, index=['reviewerID'], values=['overall'],
                            aggfunc={'overall': [len, np.min, np.mean, np.median, np.max]})
conv_pivot2df(cluster0_sum)
cluster0_sum['reviewerID'] = cluster0_sum.index

print(cluster0_sum.describe())

cluster0_review_dist = cluster0_sum['len'].value_counts()

# =============================================================================
# 10.05.01 | Individual Recommendations | Cluster 1
# =============================================================================
cluster1_enhanced = pd.merge(cluster1_df, product_df, on='asin', how='inner')
cluster1_sub = cluster1_enhanced[['asin','reviewerID','predicted_rating','category2_t','category3_t','price_t']]
cluster1_sub = cluster1_sub.reset_index().sort_values(by=['reviewerID','predicted_rating'],ascending=False).reset_index(drop=True)
cluster1_sub = cluster1_sub.drop('index', axis=1)

# start picking predictions
# cluster1_sub['reviewerID'].head(50)
print('User 1:\n', cluster1_sub[cluster1_sub['reviewerID']=='AZZTCG8URXK9W'].to_string())

print('User 2:\n', cluster1_sub[cluster1_sub['reviewerID']=='AZYZ3TSE1N2Z5'].to_string())

print('User 3:\n', cluster1_sub[cluster1_sub['reviewerID']=='AZX4Z7YMI5C6S'].to_string())

print('User 4:\n', cluster1_sub[cluster1_sub['reviewerID']=='AZY6DXJE7S2JY'].to_string())

print('User 5:\n', cluster1_sub[cluster1_sub['reviewerID']=='AZX7GJRLMWN92'].to_string())

# the recommendations are different

cluster1_sum = cluster1_enhanced['category2_t'].value_counts()


# =============================================================================
# 10.05.02 | Cluster 1 Analysis
# =============================================================================
# let's look at user profiles  - based on what actually happened
cluster1_actual = pd.merge(cluster1_df['reviewerID'],reviews_df,on='reviewerID',how='inner')

cluster1_sum = pd.pivot_table(cluster1_actual, index=['reviewerID'], values=['overall'],
                            aggfunc={'overall': [len, np.min, np.mean, np.median, np.max]})
conv_pivot2df(cluster1_sum)
cluster1_sum['reviewerID'] = cluster1_sum.index

print(cluster1_sum.describe())

cluster1_review_dist = cluster1_sum['len'].value_counts()

# ===============================================================================
# 04.00.01 | Product Metadata Features | Documentation
# ===============================================================================
# Name:               04_product_metadata_features
# Author:             Rodd
# Last Edited Date:   10/19/19
# Description:        Create relevant metataa features from the product data set.
#                     
# Notes:              Did not process the related field b/c unclear how it will be used.
#                     Brand variable is extremely sparse so was omitted.
#                     Title and description variables have a few missing vals but were left as is.
# Warnings:
#
# Outline:            Imports needed packages and objects from other scripts.
#                     Extracts product categories and creates new category variables.
#                     Creates a field for hasDescription based on product description.
#                     Fills missing price information with 0.
#                     Extracts sales rank information for the electronics cat only.
#                     Creates relevant features from the reviews data.
#                     Creates relevant features from the qa data.
#                     Removes unneeded columns and performs joins to create final data frame.
#                     Data frame is pickled and saved to the data folder.
#                     Some features are transformed via one-hot encoding.
#                     One-hot encoded data frame is saved to the data folder.
#
#
# =============================================================================
# 04.00.02 | Import Modules & Packages
# =============================================================================
# Import packages
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import gc
import pickle

# Import modules (other scripts)
from data_load import reviews_df, metadata_df, qa_df
from environment_configuration import set_palette, set_levels 
from functions import conv_pivot2df

print('Script: 04.00.02 [Import Packages] completed')

# =============================================================================
# 04.01.01 | Create product category columns
# =============================================================================
# make a copy for data prep
product_df = metadata_df.copy()

product_reviews_combined = pd.merge(metadata_df, reviews_df, on='asin',how='inner')

# split categories variable into separate category columns - there are 6 categories in total (trial & error process to figure that out)
categories_df = pd.DataFrame(product_df['categories'].str.get(0).values.tolist(), 
             columns=['category1','category2','category3','category4','category5','category6'])

# add the new category columns back to our original data frame
product_df = pd.concat([product_df, categories_df], axis=1)


# =============================================================================
# 04.01.02 | Prepare new categories columns
# =============================================================================
# filter the data to only include electronics & cell phones & acessories category1
product_df = product_df[product_df['category1'].isin(["Electronics","Cell Phones & Accessories"])]

# if a category has a count of < 100, then those categories are combined into an 'Other' category
product_df['category2_t'] = set_levels(product_df, product_df['category2'])['category2']
product_df['category3_t'] = set_levels(product_df, product_df['category3'])['category3']
product_df['category4_t'] = set_levels(product_df, product_df['category4'])['category4']
product_df['category5_t'] = set_levels(product_df, product_df['category5'])['category5']
product_df['category6_t'] = set_levels(product_df, product_df['category6'])['category6']

# if a product does not have a category, then that will be filled in with 'Unknown'
# category2 has 100% penetration, which is why it is not listed below
product_df['category3_t'].fillna('Unknown', inplace = True)
product_df['category4_t'].fillna('Unknown', inplace = True)
product_df['category5_t'].fillna('Unknown', inplace = True)
product_df['category6_t'].fillna('Unknown', inplace = True)


# =============================================================================
# 04.02.01 | Prepare description data
# =============================================================================
# create a binary variable to determine if a product has a description or not
product_df['hasDescription'] = np.where(product_df['description'].isnull(), 0, 1)


# =============================================================================
# 04.03.01 | Prepare sale price data
# =============================================================================
price_count = product_df.groupby('price').size().to_frame('products').reset_index()
# there are some really low price values (.01) - not sure why

# fill the missing values with 0
product_df['price_t'] = product_df['price'].fillna(0)


# =============================================================================
# 04.04.01 | Prepare brand data
# =============================================================================
brand_count = product_df.groupby('brand').size().to_frame('products').reset_index()
# largest brand is Unknown at 3402
# there is also a brand of nan - plus most do not have a brand - only 142526 have a brand
# there are 9994 unique brand values - not sure it's worth one hot encoding all of these


# =============================================================================
# 04.05.01 | Prepare sales rank data
# =============================================================================
# the sales rank column is a dictionary so must be transformed so that keys are cols w/ vals
# this takes a while to run
# there are 29 categories with salesRank information but one is 0 so really 28
sales_rank_vars = product_df['salesRank'].apply(pd.Series)
gc.collect()

# grab the electronics rank and join that back to our data frame
# discarding other rank vars for now
product_df = pd.concat([product_df, sales_rank_vars['Electronics']], axis=1)

# rename the electronics column
product_df.rename(columns={'Electronics':'electronicsSalesRank'}, inplace=True)

# would be good to bin this variable b/c it's unclear how to handle unknown ranks
electronic_rank_count = product_df.groupby('electronicsSalesRank').size().to_frame('products').reset_index()
# electronic_rank_count.describe()

# create another binary variable
product_df['containsAnySalesRank'] = np.where(product_df['salesRank'].isnull(), 0, 1)


# =============================================================================
# 04.06.01 | Prepare related data
# =============================================================================
# We are missing this information for 25% of products
# And, not every product has the same related information

# skipping this on 10/19/19




# =============================================================================
# 04.07.01 | Create reviews features at a product level
# =============================================================================
# merge product and reviews data
product_reviews_combined = pd.merge(product_df, reviews_df, on='asin',how='inner')

product_review_pivot = pd.pivot_table(product_reviews_combined, index=['asin'], values=['overall'],
                              aggfunc={'overall': [len, np.mean]})

# Convert Review Pivot to Data Frame
conv_pivot2df(product_review_pivot)

product_review_pivot.rename(columns={'len': 'numberReviews', 'mean': 'meanStarRating'}, inplace=True)

# want to create variables for number of reviews with 1-star, 2-star, etc. reviews
rating_dist = product_reviews_combined.groupby(['asin','overall']).size().to_frame('reviews').reset_index()
rating_dist = pd.merge(rating_dist, product_review_pivot['numberReviews'], on='asin',how='inner')
rating_dist['ratingProp'] = rating_dist['reviews']/rating_dist['numberReviews']

# recode overall to categorical
rating_vals = {1: 'star1Rating', 2: 'star2Rating', 3: 'star3Rating',4: 'star4Rating',5: 'star5Rating'} 
rating_dist['overall'] = [rating_vals[item] for item in rating_dist['overall']] 

# remove unncessary columns
rating_dist = rating_dist.drop(['numberReviews','reviews'], axis=1)

# now, we spread to create separate columns for prop of ratings in each star level
rating_dist = pd.pivot_table(rating_dist,index='asin',columns='overall',values='ratingProp')

# fill missing vals with 0
rating_dist['star1Rating'] = rating_dist['star1Rating'].fillna(0)
rating_dist['star2Rating'] = rating_dist['star2Rating'].fillna(0)
rating_dist['star3Rating'] = rating_dist['star3Rating'].fillna(0)
rating_dist['star4Rating'] = rating_dist['star4Rating'].fillna(0)
rating_dist['star5Rating'] = rating_dist['star5Rating'].fillna(0)


# =============================================================================
# 04.08.01 | Add in Q/A data
# =============================================================================
# first we summarize this data so that it is easier to join into
# upon inspecting the data, the number of questions asked and answers given is the same

# get the number of questions by product
question_sum = qa_df.groupby('asin').agg({'question': 'count'}).rename(columns={'question':'numberQuestions'}).reset_index()

# join this back to our product_df
product_df = pd.merge(product_df, question_sum, on='asin',how='left')

# fill missing vals with 0
product_df['numberQuestions'] = product_df['numberQuestions'].fillna(0)

# change data type to int
product_df['numberQuestions'] = product_df['numberQuestions'].astype(int)


# =============================================================================
# 04.09.01 | Drop columns
# =============================================================================
# verify data types and make changes if needed
# product_df.dtypes

# want to create a full data frame 
# this is for safe-keeping
product_df_full = product_df


# category1 is not needed b/c was used for filtering
# do not need to keep the original categories column
# cannot use the URL for modeling - not in scope
# brand variable has very low penetration and extreme variability
# salesRank was transformed
# price had missing values filled in
# also removing related for now
# original category vars are also removed
columns_to_remove = ['category1','categories','imUrl','brand','salesRank','price','related',
                     'category2','category3','category4','category5','category6']
product_df = product_df.drop(columns_to_remove, axis=1)

product_df_sub = product_df


# =============================================================================
# 04.10.01 | Join to reviews data for full feature set
# =============================================================================
# join to reviews and only keep the product info that we need
product_final = pd.merge(product_df, product_review_pivot, on='asin', how='inner')
product_final = pd.merge(product_final, rating_dist, on='asin', how='left')


# =============================================================================
# 04.11.01 | Create binned var for electronics rank
# =============================================================================
# this step will help with missing values b/c can't just set these to a certain val
# product_final['electronicsSalesRank'].describe()

# use percentiles to create bins
# the bins include the top number so it would be (0, 66682]
product_final['electronicsRankBin'] = pd.cut(x=product_final['electronicsSalesRank'], 
          bins=[0, 8541.75,32031.50,69008.75,810712.],
          labels=['25thPercentile', '50thPercentile', '75thPercentile', '100thPercentile'])

# have to replace missing values with Unknown
product_final['electronicsRankBin'] = product_final['electronicsRankBin'].cat.add_categories(["Unknown"])
product_final['electronicsRankBin'] = product_final['electronicsRankBin'].fillna("Unknown")
product_final['electronicsRankBin'] = product_final['electronicsRankBin'].astype(str)

# remove the original Electronics Sales Rank var
product_final = product_final.drop('electronicsSalesRank', axis=1)


# =============================================================================
# 04.10.01 | Save data set
# =============================================================================
product_final.to_pickle("C:\\Users\\julia\\OneDrive\\Documents\\Code\\capstone\\data\\product_metadata_no_one_hot_encoding.pkl")

# df = pd.read_pickle("C:\\Users\\julia\\OneDrive\\Documents\\Code\\capstone\\data\\product_metadata_no_one_hot_encoding.pkl")


# =============================================================================
# 04.11.01 | Create One-Hot Encoding for electronicsRankBin
# =============================================================================
product_one_hot = product_final

# before we create dummy vars, want to rename the values so that columns are descriptive when spread
product_one_hot['electronicsRankBin'] = 'electronicsRank' + product_one_hot['electronicsRankBin']

# verify levels
product_one_hot['electronicsRankBin'].unique()

# get one hot encoding of electronicsRankBin
# this approach is used over scikit-learn b/c we want column names to be descriptive
# we do not not want electronicsRank1, electronicsRank2, etc.
electronics_one_hot = pd.get_dummies(product_one_hot['electronicsRankBin'])

# join the encoded df
product_one_hot = product_one_hot.join(electronics_one_hot)

# drop electronicsRankBin since it is now encoded
product_one_hot = product_one_hot.drop('electronicsRankBin',axis = 1)


# =============================================================================
# 04.12.01 | Create One-Hot Encoding for category2_t
# =============================================================================
# product_one_hot['category2_t'].nunique() - 14
# product_one_hot['category3_t'].nunique() - 88
# only doing this for category2 b/c do not want to bloat the data

# before we create dummy vars, want to rename the values so that columns are descriptive when spread
# also removing white spaces
product_one_hot['category2_t'] = 'category2' + product_one_hot['category2_t'].str.replace(' ', '')

# get one hot encoding of category2_t
cat2_one_hot = pd.get_dummies(product_one_hot['category2_t'])

# join the encoded df
product_one_hot = product_one_hot.join(cat2_one_hot)

# drop category2_t since it is now encoded
product_one_hot = product_one_hot.drop('category2_t',axis = 1)


# =============================================================================
# 04.13.01 | Drop Other Columns
# =============================================================================
columns_to_remove = ['category3_t','category4_t','category5_t','category6_t']
product_one_hot = product_one_hot.drop(columns_to_remove, axis=1)


# =============================================================================
# 04.14.01 | Save one-hot encoded data set
# =============================================================================
product_one_hot.to_pickle("C:\\Users\\julia\\OneDrive\\Documents\\Code\\capstone\\data\\product_metadata_one_hot_encoding.pkl")


# =============================================================================
# APPENDIX
# =============================================================================
# =============================================================================
# 04.01.02 | Inspect new categories columns
# =============================================================================
#cat1_count = product_df.groupby('category1').size().to_frame('products').reset_index()
# why are there some categories outside of electronices? is this in error?

#clothing = product_df[product_df['category1']=='Clothing, Shoes & Jewelry'] # these are like kindle cases
#baby = product_df[product_df['category1']=='Baby Products']
#books = product_df[product_df['category1']=='Books'] # looks like books for kindle....

# category1 is not a helpful field to use - but should be used for filtering to remove some edge cases

#cat2_count = product_df[product_df['category1'].isin(["Electronics","Cell Phones & Accessories"])].groupby('category2').size().to_frame('products').reset_index()
# there are several categories with only 1 or 2 products - need to combine all of those into an 'Other' category

#cat3_count = product_df[product_df['category1'].isin(["Electronics","Cell Phones & Accessories"])].groupby('category3').size().to_frame('products').reset_index()
#cat4_count = product_df[product_df['category1'].isin(["Electronics","Cell Phones & Accessories"])].groupby('category4').size().to_frame('products').reset_index()
#cat5_count = product_df[product_df['category1'].isin(["Electronics","Cell Phones & Accessories"])].groupby('category5').size().to_frame('products').reset_index()
#cat6_count = product_df[product_df['category1'].isin(["Electronics","Cell Phones & Accessories"])].groupby('category6').size().to_frame('products').reset_index()

#product_df['category2'].nunique()
#Out[70]: 33
#
#product_df['category3'].nunique()
#Out[71]: 138
#
#product_df['category4'].nunique()
#Out[72]: 321
#
#product_df['category5'].nunique()
#Out[73]: 304
#
#product_df['category6'].nunique()
#Out[74]: 79

#498132 is the number of unique products from electronics and cell phones & accessories
#print('category2 penetration: ',sum(cat2_count['products'])/498132)
#print('category3 penetration: ',sum(cat3_count['products'])/498132)
#print('category4 penetration: ',sum(cat4_count['products'])/498132)
#print('category5 penetration: ',sum(cat5_count['products'])/498132)
#print('category6 penetration: ',sum(cat6_count['products'])/498132)

# count the number of NaN values in each column
#print(product_df.isnull().sum())
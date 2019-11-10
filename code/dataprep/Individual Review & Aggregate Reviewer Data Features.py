# ===============================================================================
# 04.00.01 | Individual Review & Aggregated Reviewer Features | Documentation
# ===============================================================================
# Name:               Individual Review & Aggregated Reviewer Features
# Author:             Hemant Patel
# Last Edited Date:   10/26/19
# Description:        Create relevant review and aggregate reviewer features from the reviews dataset.
#                     
# Notes:                              
# Warnings:
# Outline:            Imports needed packages and objects.
#                     Extracts reviews data and creates relevant features at the individual review level.
#                     Appends product metadata to enhance and provide additional contenxt to review data.
#                     Creates aggregated reviewer level data and relevant features at reviewer level.
#                     Removes unneeded columns where appropriate and create final output dataframes.
#                     Creates pickled files which are then and saved to the data folder in github.
#
# =============================================================================
# 04.00.02 | Import Modules & Packages
# =============================================================================

# Import packages
import pandas as pd
import numpy as np
import pickle

# =============================================================================
# 04.00.03 | Import Data
# =============================================================================

# Load pickled review data where reviewer left at least 5 reviews
with open ('./reviewsdata_5.pkl', 'rb') as pickle_file:
    reviewdata = pickle.load(pickle_file)

# Load pickled metadata
with open ('./product_metadata_no_one_hot_encoding.pkl', 'rb') as pickle_file:
    metadata = pickle.load(pickle_file)

# Count number of records for both datasets
print('Reviews Record Count:', len(reviewdata))
print('Metadata Record Count:', len(metadata))

# =============================================================================
# 04.00.04 | Create new reviews dataframe containing only relevant fields
# =============================================================================

# Removed "reviewerName" -> not useful for modeling
# Removed "unixReviewTime" -> use "reviewTime" instead
reviewdata_reduced = reviewdata[['reviewerID', 'asin','helpful','reviewText','overall','summary','reviewTime']]

# =============================================================================
# 04.00.05 | Split "helpful" into numerator and denominator
# =============================================================================

helpful_numer_df = pd.DataFrame(reviewdata_reduced['helpful'].str.get(0).values.tolist(), columns=['helpful_numer'])
helpful_denom_df = pd.DataFrame(reviewdata_reduced['helpful'].str.get(1).values.tolist(), columns=['helpful_denom'])

# =============================================================================
# 04.00.06 | Add "helpful numerator" and "helpful denominator" to original dataframe
# =============================================================================

reviewdata_reduced = pd.concat([reviewdata_reduced, helpful_numer_df], axis=1)
reviewdata_reduced = pd.concat([reviewdata_reduced, helpful_denom_df], axis=1)

# =============================================================================
# 04.00.07 | Create flag indicating if "helpful" is greater than 0
# =============================================================================

reviewdata_reduced.loc[reviewdata_reduced.helpful_denom < 1, 'helpful_flag'] = 0
reviewdata_reduced.loc[reviewdata_reduced.helpful_denom >= 1, 'helpful_flag'] = 1

# =============================================================================
# 04.00.08 | Compute proportion of "helpful_numer" and "helpful_denom"
# =============================================================================

reviewdata_reduced.loc[reviewdata_reduced.helpful_flag == 1, 'helpful_proportion'] = reviewdata_reduced['helpful_numer']/reviewdata_reduced['helpful_denom']
reviewdata_reduced.loc[reviewdata_reduced.helpful_flag == 0, 'helpful_proportion'] = 0

# =============================================================================
# 04.00.09 | Convert "reviewTime" to date format of YEAR-MON-DAY
# =============================================================================

reviewdata_reduced['reviewDate'] = pd.to_datetime(reviewdata_reduced['reviewTime'], format='%m %d, %Y')
reviewdata_reduced['reviewYear'] = reviewdata_reduced['reviewDate'].dt.year
reviewdata_reduced['reviewMonth'] = reviewdata_reduced['reviewDate'].dt.month

# =============================================================================
# 04.00.10 | Drop unnecessary columns like "helpful" and "reviewTime"
# =============================================================================

reviewdata_reduced = reviewdata_reduced.drop(['helpful', 'reviewTime'], axis=1)

# =============================================================================
# 04.00.11 | Append review dataframe with characteristics from metadata
# =============================================================================

reviews_meta_combined = pd.merge(reviewdata_reduced, metadata, on='asin',how='inner')
print(len(reviews_meta_combined))

# =============================================================================
# 04.00.12 | Drop unnecessary columns
# =============================================================================

reviews_meta_combined_reduced = reviews_meta_combined.drop(['numberReviews','star1Rating','star2Rating',
                                                            'star3Rating','star4Rating','star5Rating'], axis=1)

# =============================================================================
# 04.00.13 | Create flag indicating if individual review rating is greater than average rating
# =============================================================================

reviews_meta_combined_reduced.loc[reviews_meta_combined_reduced.overall >= reviews_meta_combined_reduced.meanStarRating, 'overall_rating_flag'] = 1
reviews_meta_combined_reduced.loc[reviews_meta_combined_reduced.overall < reviews_meta_combined_reduced.meanStarRating, 'overall_rating_flag'] = 0

# =============================================================================
# 04.00.14 | Create summary level pivot table for ratings
# =============================================================================

# Function to convert pivot table to dataframe
def conv_pivot2df(pivot_name):
    '''Pandas pivot table to data frame; requires pandas'''
    pivot_name.columns = pivot_name.columns.droplevel(0)
    pivot_name = pivot_name.reset_index().rename_axis(None, axis=1)
    pivot_name = pd.DataFrame(pivot_name)

# For each unique reviewer count number of reviews, compute rating average, find min & median & max of rating values
rating_pivot = pd.pivot_table(reviews_meta_combined_reduced, index=['reviewerID'], values=['overall'],
                            aggfunc={'overall': [len, np.min, np.mean, np.median, np.max, np.sum]})

# Convert rating pivot to dataframe
conv_pivot2df(rating_pivot)
rating_pivot = rating_pivot.reset_index()


# =============================================================================
# 04.00.15 | Create summary level pivot tables for price
# =============================================================================

# For each unique reviewer count number of reviews, compute average price, find min & median & max price
price_pivot = pd.pivot_table(reviews_meta_combined_reduced, index=['reviewerID'], values=['price_t'],
                            aggfunc={'price_t': [len, np.min, np.mean, np.median, np.max, np.sum]})

# Convert price pivot to dataframe
conv_pivot2df(price_pivot)
price_pivot = price_pivot.reset_index()

# =============================================================================
# 04.00.16 | Create summary level pivot table for helpfulness numerator
# =============================================================================

# For each unique reviewer count number of reviews and aggregate helpful numerator values
helpful_numer_pivot = pd.pivot_table(reviews_meta_combined_reduced, index=['reviewerID'], values=['helpful_numer'],
                            aggfunc={'helpful_numer': [len, np.sum]})

# Convert helpful numerator pivot to dataframe
conv_pivot2df(helpful_numer_pivot)
helpful_numer_pivot = helpful_numer_pivot.reset_index()

# =============================================================================
# 04.00.17 | Create summary level pivot table for helpfulness denominator
# =============================================================================

# For each unique reviewer count number of reviews and aggregate helpful denominator values
helpful_denom_pivot = pd.pivot_table(reviews_meta_combined_reduced, index=['reviewerID'], values=['helpful_denom'],
                            aggfunc={'helpful_denom': [len, np.sum]})

# Convert helpful denominator pivot to dataframe
conv_pivot2df(helpful_denom_pivot)
helpful_denom_pivot = helpful_denom_pivot.reset_index()

# =============================================================================
# 04.00.18 | Create summary level pivot table for number of days between each review executed
# =============================================================================

# Sort data by ID and review date
reviews_meta_combined_reduced = reviews_meta_combined_reduced.sort_values(by=['reviewerID', 'reviewDate'])

# Compute the number of days between each review executed 
reviews_meta_combined_reduced['reviewDate_diff'] = reviews_meta_combined_reduced.groupby('reviewerID')['reviewDate'].diff(-1) * (-1)
reviews_meta_combined_reduced['reviewDate_diff'] = reviews_meta_combined_reduced['reviewDate_diff'].dt.days
reviews_meta_combined_reduced['reviewDate_diff'].fillna(0, inplace=True)
reviews_meta_combined_reduced.head()

# For each unique reviewer count number of reviews, calculate average duration, find min & median & max duration
reviewDate_diff_pivot = pd.pivot_table(reviews_meta_combined_reduced, index=['reviewerID'], values=['reviewDate_diff'],
                            aggfunc={'reviewDate_diff': [len, np.min, np.mean, np.median, np.max, np.sum]})

# Convert data duration pivot to dataframe
conv_pivot2df(reviewDate_diff_pivot)
reviewDate_diff_pivot = reviewDate_diff_pivot.reset_index()

# =============================================================================
# 04.00.19 | Rename column names across all pivot tables and combine into singlular dataframe
# =============================================================================

# Ratings pivot rename
rating_pivot = rating_pivot.rename(columns = {"amax":"MaxRating", 
                                              "amin":"MinRating",
                                              "len":"NumberOfRatings", 
                                              "mean":"AverageRating",
                                              "median":"MedianRating",
                                              "sum":"SummedRatings"})

# Price pivot rename
price_pivot = price_pivot.rename(columns = {"amax":"MaxPrice",
                                            "amin":"MinPrice",
                                            "len":"NumberOfPrice", 
                                            "mean":"AveragePrice",
                                            "median":"MedianPrice",
                                            "sum":"SummedPrice"})

# Helpful numerator pivot rename
helpful_numer_pivot = helpful_numer_pivot.rename(columns = {"len":"NumberOfHelpfulNumer",
                                                            "sum":"SummedHelpfulNumer"})

# Helpful denominator pivot rename
helpful_denom_pivot = helpful_denom_pivot.rename(columns = {"len":"NumberOfHelpfulDenom",
                                                            "sum":"SummedHelpfulDenom"})

# Review date duration pivot rename
reviewDate_diff_pivot = reviewDate_diff_pivot.rename(columns = {"amax":"MaxNumDaysBetweenReviews",
                                                                "amin":"MinNumDaysBetweenReviews",
                                                                "len":"NumberOfReviewDates",
                                                                "mean":"AverageNumDaysBetweenReviews",
                                                                "median":"MedianNumDaysBetweenReviews",
                                                                "sum":"SummedNumDaysBetweenReviews"})

# Combine individual pivot tables into singular dataframe
join_1 = pd.merge(rating_pivot, price_pivot, on='reviewerID',how='inner')
join_2 = pd.merge(join_1, helpful_numer_pivot, on='reviewerID',how='inner')
join_3 = pd.merge(join_2, helpful_denom_pivot, on='reviewerID',how='inner')
reviews_meta_combined_aggregated = pd.merge(join_3, reviewDate_diff_pivot, on='reviewerID',how='inner')
reviews_meta_combined_aggregated.head()

# =============================================================================
# 04.00.20 | Drop unnecessary columns
# =============================================================================

reviews_meta_combined_aggregated = reviews_meta_combined_aggregated.drop(['NumberOfPrice',
                                                                          'NumberOfHelpfulNumer',
                                                                          'NumberOfHelpfulDenom',
                                                                          'NumberOfReviewDates'],axis=1)

# =============================================================================
# 04.00.21 | Pickle individual review level data and aggregated reviewer level data
# =============================================================================

# Create copy of dataframe for export
reviews_meta_combined_individual = reviews_meta_combined_reduced

# Pickle individual review level data
reviews_meta_combined_individual.to_pickle('./reviews_meta_combined_individual.pkl')

# Pickle aggregated reviewer level data
reviews_meta_combined_aggregated.to_pickle('./reviews_meta_combined_aggregated.pkl')

# =============================================================================
# 
# =============================================================================
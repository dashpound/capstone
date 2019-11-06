# ===============================================================================
# 07.00.01 | Paper Visuals | Documentation
# ===============================================================================
# Name:               07_product_metadata_features
# Author:             Rodd
# Last Edited Date:   11/5/19
# Description:        Create EDA visuals to use in papers.
#                     
# Notes:              Have not saved these visuals to the output folder - that is a next level enhancement.
#                     
#
# Warnings:           Did not work to clean up the blank plot that is generated with some of the visuals.
#
#
# Outline:            Imports needed packages and objects from other scripts & set plot params and palette.
#                     Create a combined product and reviews data frame for visualization.
#                     Create a reviews by rating visual for Data Overview section.
#                     Prepare top-level category data for visualization.
#                     Generate reviews and products plots using top-level category.
#                     Perform data prep to show product data missingness.
#                     Create plot for product data missingness.
#                     Create a category 2 plot by products.
#                     Create a plot for average rating for each product.
#                     Generate average ratings violin plot showed in project goals document.
#
#
# ===============================================================================
# 07.00.02 | Import packages and define global params
# ===============================================================================
# Import packages
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import numpy as np
import pandas as pd
import gc

# Import modules (other scripts)
from data_load import reviews_df, metadata_df
from clean_data_load import products_clean
from environment_configuration import show_values_on_bars, plot_params
from eda import base_pivot

# need to call plot_params
plot_params()

# defining color palette this way so we can more readily pick off colors
my_palette = sns.color_palette()

print('Script: 07.00.02 [Import Packages] completed')

# =============================================================================
# Initial Findings Visuals
# =============================================================================
# =============================================================================
# 07.01.01 | Create product and reviews data frame
# =============================================================================
# make a copy for data prep
product_df = metadata_df.copy()

# split categories variable into separate category columns - there are 6 categories in total (trial & error process to figure that out)
categories_df = pd.DataFrame(product_df['categories'].str.get(0).values.tolist(), 
             columns=['category1','category2','category3','category4','category5','category6'])

# add the new category columns back to our original data frame
product_df = pd.concat([product_df, categories_df], axis=1)

# join this to the reviews data set so we can generate plots using the category
product_reviews_combined = pd.merge(product_df, reviews_df, on='asin',how='inner')

print('Script: 07.01.01 [Product and reviews data] completed')

# =============================================================================
# 07.02.01 | Reviews by Rating Visual
# =============================================================================
review_byrating_sum = product_reviews_combined.groupby('overall').size().to_frame('reviews').reset_index()

fig, ax = plt.subplots(1, 1, figsize = (8,4))
ax.set_ylim(1, 5) # sets start of y-axis to 1 instead of 0
ax = review_byrating_sum.plot("overall", "reviews", kind="barh", color = my_palette[0], legend=None)
plt.title('Reviews by Product Rating',fontweight='bold')  
plt.xlabel('Number of Reviews',fontweight='bold')
plt.ylabel('Product Rating',fontweight='bold')
# format axis so that thousands show up with a K
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: '{:,.0f}'.format(x/1000) + 'K'))
# add percentage labels to plot
total = sum(review_byrating_sum['reviews'])
for p in ax.patches:
        percentage = '{:.0f}%'.format(100 * p.get_width()/total)
        x = p.get_x() + p.get_width() + 0.02
        y = p.get_y() + p.get_height()/2
        ax.annotate(percentage, (x, y))
plt.tight_layout()
plt.show()

print('Script: 07.02.01 [Reviews by product rating visual] completed')


# =============================================================================
# 07.03.01 | Top Level Category Data Frames
# =============================================================================
cat1_sum = product_reviews_combined.groupby('category1').size().to_frame('reviews').reset_index().sort_values('reviews', ascending=False)
cat1_product_sum = product_reviews_combined.groupby('category1').aggregate({'asin': pd.Series.nunique}).reset_index().sort_values('asin', ascending=False)

# need to clean this up for plotting
cat1_sum['category1'] = cat1_sum['category1'].map({'Cell Phones & Accessories': 'Cell Phones &\n Accessories',
                                                   'Clothing, Shoes & Jewelry': 'Clothing,\nShoes & Jewelry',
                                                   'Electronics': 'Electronics', 'Automotive':'Automotive'})


cat1_product_sum = product_reviews_combined.groupby('category1').aggregate({'asin': pd.Series.nunique}).reset_index().sort_values('asin', ascending=False)

# need to clean this up for plotting
cat1_product_sum['category1'] = cat1_product_sum['category1'].map({'Cell Phones & Accessories': 'Cell Phones &\n Accessories',
                                                   'Clothing, Shoes & Jewelry': 'Clothing,\nShoes & Jewelry',
                                                   'Electronics': 'Electronics', 'Automotive':'Automotive'})

print('Script: 07.03.01 [Top-level category data prep] completed')


# =============================================================================
# 07.03.02 | Top Level Category Plots
# =============================================================================
fig, axes = plt.subplots(nrows=1, ncols=2, figsize = (15,4))

# top level category by product
ax1 = axes[0]
# define the plot
sns_b = sns.barplot(data=cat1_product_sum, x="category1", y="asin", color = my_palette[2],ax=ax1) 
# still need to add fontweight of 'bold' here
ax1.set_title('Products by Top Level Product Category\n',fontweight='bold')  
ax1.set_xlabel('',fontweight='bold')
ax1.set_ylabel('Products',fontweight='bold')
# add labels to plot
show_values_on_bars(sns_b, "v")
# set axis labels
ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: '{:,.0f}'.format(x/1000) + 'K'))

# top level category by review
ax2 = axes[1]
# define the plot
sns_b = sns.barplot(data=cat1_sum, x="category1", y="reviews", color = my_palette[0],ax=ax2) 
# need to add more space to title from plot
ax2.set_title('Reviews by Top Level Product Category\n',fontweight='bold') 
# still need to add fontweight of 'bold' here 
ax2.set_xlabel('',fontweight='bold')
ax2.set_ylabel('Reviews',fontweight='bold')
# add labels to plot
show_values_on_bars(sns_b, "v")
# set axis labels
ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: '{:,.0f}'.format(x/1000) + 'K'))

print('Script: 07.03.02 [Top-level category plots] completed')


# =============================================================================
# 07.04.01 | Product Data Missingness Data Prep
# =============================================================================
product_df = metadata_df.copy()

# first join to reviews data to get the appropriate product scope
product_tmp = pd.merge(product_df, reviews_df[['asin']], on='asin',how='inner')

product_tmp = product_tmp['asin'].drop_duplicates()

# now join back to the product data
product_df2 = pd.merge(product_df, product_tmp, on='asin',how='inner')

# now we can spread some variables to accurately report on missingness
# split categories variable into separate category columns - there are 6 categories in total (trial & error process to figure that out)
categories_df = pd.DataFrame(product_df2['categories'].str.get(0).values.tolist(), 
             columns=['category1','category2','category3','category4','category5','category6'])

# add the new category columns back to our original data frame
product_df2 = pd.concat([product_df2, categories_df], axis=1)

product_df2 = product_df2.drop('categories', axis=1)

sales_rank_vars = product_df2['salesRank'].apply(pd.Series)
gc.collect()

# grab the electronics rank and join that back to our data frame
# discarding other rank vars for now
product_df2 = pd.concat([product_df2, sales_rank_vars['Electronics']], axis=1)

# rename the electronics column
product_df2.rename(columns={'Electronics':'electronicsSalesRank'}, inplace=True)

print('Script: 07.04.01 [Product data missingness data prep] completed')


# =============================================================================
# 07.04.02 | Product Data Missingness Plot
# =============================================================================
total_products = product_df2['asin'].nunique()
missing_sum = product_df2.isnull().sum().to_frame('numberMissing').reset_index().sort_values('numberMissing', ascending=True)

fig, ax = plt.subplots(1, 1, figsize = (8,4))
ax = missing_sum.plot("index", "numberMissing", kind="barh", color = my_palette[2], legend=None)
plt.title('Product Metadata Missingness',fontweight='bold')  
plt.xlabel('Number of Products',fontweight='bold')
plt.ylabel('',fontweight='bold')
# format axis so that thousands show up with a K
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: '{:,.0f}'.format(x/1000) + 'K'))
for p in ax.patches:
        percentage = '{:.0f}%'.format(100 * p.get_width()/total_products)
        x = p.get_x() + p.get_width() + 0.02
        y = p.get_y() + p.get_height()/2
        ax.annotate(percentage, (x, y))
plt.tight_layout()
plt.show()

print('Script: 07.04.02 [Product data missingness plot] completed')


# =============================================================================
# 07.05.01 | Category 2 Distribution
# =============================================================================
product_cat2_sum = product_reviews_combined.groupby('category2_t').aggregate({'asin': pd.Series.nunique}).reset_index().sort_values('asin', ascending=True)

fig, ax = plt.subplots(1, 1, figsize = (8,4))
ax = product_cat2_sum.plot("category2_t", "asin", kind="barh", color = my_palette[2], legend=None)
plt.title('Products by Category 2',fontweight='bold')  
plt.xlabel('Number of Products',fontweight='bold')
plt.ylabel('',fontweight='bold')
# format axis so that thousands show up with a K
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: '{:,.0f}'.format(x/1000) + 'K'))
# add percentage labels to plot
total = sum(product_cat2_sum['asin'])
for p in ax.patches:
        percentage = '{:.0f}%'.format(100 * p.get_width()/total)
        x = p.get_x() + p.get_width() + 0.02
        y = p.get_y() + p.get_height()/2
        ax.annotate(percentage, (x, y))
plt.tight_layout()
plt.show()

print('Script: 07.04.01 [Category 2 distribution plot] completed')


# =============================================================================
# 07.06.01 | Products by Average Rating
# =============================================================================
product_avgrating_sum = products_clean.groupby('meanStarRating').aggregate({'asin': pd.Series.nunique}).reset_index().sort_values('asin', ascending=True)

fig, ax = plt.subplots(1, 1, figsize = (8,4))
ax = sns.distplot(products_clean['meanStarRating'], color = my_palette[2]) 
plt.title('Average Product Rating for Each Product',fontweight='bold')  
plt.xlabel('Average Rating',fontweight='bold')
plt.ylabel('',fontweight='bold')
plt.tight_layout()
plt.show()

print('Script: 07.06.01 [Products by average rating] completed')


# =============================================================================
# 07.07.01 | Side by Side Distribution Plots | Executive Summary
# =============================================================================
# Convert to Data Frame
from functions import conv_pivot2df
conv_pivot2df(base_pivot)

# rename column
base_pivot.rename(columns={'mean':'meanStarRating'}, inplace=True)
# wanted to pull out reviewerID into a column for flexibility
base_pivot.reset_index(level=0, inplace=True)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize = (15,4))

# top level category by product
ax1 = axes[0]
# define the plot
sns_h = sns.distplot(base_pivot['meanStarRating'], color = my_palette[0], ax=ax1) 
# still need to add fontweight of 'bold' here
ax1.set_title('Average Product Rating for Each Reviewer',fontweight='bold')  
ax1.set_xlabel('Average Rating',fontweight='bold')
ax1.set_ylabel('',fontweight='bold')


# top level category by review
ax2 = axes[1]
# define the plot
sns_h2 = sns.distplot(products_clean['meanStarRating'], color = my_palette[2], ax=ax2) 
ax2.set_title('Average Product Rating for Each Product',fontweight='bold')  
ax2.set_xlabel('Average Rating',fontweight='bold')
ax2.set_ylabel('',fontweight='bold')
plt.tight_layout()
plt.show()

print('Script: 07.07.01 [Histograms of ratings for executive summary] completed')


# =============================================================================
# Project Goals Visuals
# =============================================================================
# =============================================================================
# 07.08.01 | Violin Plot of Avg Ratings
# =============================================================================
avg_rating_by_reviewer = base_pivot.groupby('mean').count().reset_index()
# verified sum(avg_rating_by_reviewer['len']) == 192403

# set plot params for styling
fig, ax = plt.subplots(1, 1, figsize = (8,4))
# define the plot
sns.violinplot(x = base_pivot["mean"], color=my_palette[1]) 
plt.title('Average Product Rating Distribution by Reviewer',fontweight='bold') 
plt.xlabel('Rating')
plt.xticks(np.arange(1, 6, 1)) # must be max+1

print('Script: 07.08.01 [Violin plot of average ratings for project goals] completed')


fig, ax = plt.subplots(1, 1, figsize = (8,4))

# define the plot
sns_h = sns.distplot(base_pivot['meanStarRating'], color = my_palette[0]) 
# still need to add fontweight of 'bold' here
plt.title('Average Product Rating for Each Reviewer',fontweight='bold')  
plt.xlabel('Average Rating',fontweight='bold')
plt.ylabel('',fontweight='bold')
plt.tight_layout()
plt.show()
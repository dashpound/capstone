# Import packages
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import numpy as np
import pandas as pd
import gc
import pickle

# Import modules (other scripts)
from data_load import reviews_df, metadata_df, qa_df
from environment_configuration import set_palette, set_levels, show_values_on_bars, plot_params
from functions import conv_pivot2df


# =============================================================================
# 05.01.01 | Create product and reviews data frame
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


# =============================================================================
# 05.02.01 | Reviews by Rating Visual
# =============================================================================
review_byrating_sum = product_reviews_combined.groupby('overall').size().to_frame('reviews').reset_index()

my_palette = sns.color_palette()

fig, ax = plt.subplots(1, 1, figsize = (8,4))
ax.set_ylim(1, 5) # sets start of y-axis to 1 instead of 0
ax = review_byrating_sum.plot("overall", "reviews", kind="barh", color = my_palette[0], legend=None)
plt.title('Reviews by Product Rating',fontweight='bold')  
plt.xlabel('Number of Reviews',fontweight='bold')
plt.ylabel('Product Rating',fontweight='bold')
# format axis so that thousands show up with a K
xlabels = ['{:,.0f}'.format(x) + 'K' for x in ax.get_xticks()/1000]
ax.set_xticklabels(xlabels)
# add percentage labels to plot
total = sum(review_byrating_sum['reviews'])
for p in ax.patches:
        percentage = '{:.0f}%'.format(100 * p.get_width()/total)
        x = p.get_x() + p.get_width() + 0.02
        y = p.get_y() + p.get_height()/2
        ax.annotate(percentage, (x, y))
plt.tight_layout()
plt.show()


# =============================================================================
# 05.03.01 | Top Level Category Data Frames
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

# =============================================================================
# 05.03.02 | Top Level Category Plots
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


# =============================================================================
# 05.04.01 | Product Data Missingness
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

total_products = product_df2['asin'].nunique()
missing_sum = product_df2.isnull().sum().to_frame('numberMissing').reset_index().sort_values('numberMissing', ascending=True)

fig, ax = plt.subplots(1, 1, figsize = (8,4))
ax = missing_sum.plot("index", "numberMissing", kind="barh", color = my_palette[2], legend=None)
plt.title('Product Metadata Missingness',fontweight='bold')  
plt.xlabel('Number of Products',fontweight='bold')
plt.ylabel('',fontweight='bold')
# format axis so that thousands show up with a K
#xlabels = ['{:,.0f}'.format(x) + 'K' for x in ax.get_xticks()/1000]
#ax.set_xticklabels(xlabels)
# add percentage labels to plot
for p in ax.patches:
        percentage = '{:.0f}%'.format(100 * p.get_width()/total_products)
        x = p.get_x() + p.get_width() + 0.02
        y = p.get_y() + p.get_height()/2
        ax.annotate(percentage, (x, y))
plt.tight_layout()
plt.show()



# =============================================================================
# 05.06.01 | Violin Plot of Avg Ratings - From Project Goals
# =============================================================================
# THIS STILL HAS TO BE CLEANED UP!
avg_rating_by_reviewer = base_pivot.groupby('mean').count().reset_index()
# verified sum(avg_rating_by_reviewer['len']) == 192403

# set plot params for styling
fig, ax = plt.subplots(1, 1, figsize = (8,4))
sns.set(font_scale=1.5)
sns.set_style("whitegrid")
plt.gca().xaxis.grid(True)
plt.rcParams['font.weight'] = "bold"
plt.rcParams['font.sans-serif'] = "Calibri"
# define the plot
sns.violinplot(x = base_pivot["mean"], color=my_palette[1]) 
#sns.lmplot(x="mean", y="len", data=avg_rating_by_reviewer)
plt.title('Average Product Rating Distribution by Reviewer',fontweight='bold') # bolding isn't working on title for some reason
plt.xlabel('Rating')
plt.xticks(np.arange(1, 6, 1)) # must be max+1
#plt.ylabel('Number of Unique Reviewers')
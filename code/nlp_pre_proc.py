# ===============================================================================
# 04.00.01 | nlp_pre_proc | Documentation
# ===============================================================================
# Name:               nlp_pre_proc
# Author:             Kiley
# Last Edited Date:   10/14/19
# Description:        Call data frames from EDA code;
#                     Tokenize dataframes
# Notes:
# Warnings:
#
# Outline:
#
# =============================================================================
# 04.00.02 | Import Modules & Packages
# =============================================================================
# Import packages
import re, string
import multiprocessing
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
import os
import io
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
import json_lines
import pandas as pd

# Import modules (other scripts)
from data_load import reviews_df
from environment_configuration import set_palette
from functions import clean_docs
from functions import gen_jlines
from environment_configuration import RANDOM_SEED
from functions import cluster_and_plot

print('Script: 04.00.02 [Import Packages] completed')

# =============================================================================
# 04.01.01 | Settings for sampling to facilitate development
# =============================================================================
# Generate jsonlines file, configure to n to skip
# Note: This really only needs to be run once & then stored as part of teh repository
# Default to 'n"
create_jlines = 'n'

# Sample it makes it so that the data frame is sampled for quicker development
# Configure to 'n' in production
sampleit = 'n'
# Number of records to sample if sampleit is 'y'

num_2_samp = 10000

#Set number of clusters
k = 10

# Sampling function
if sampleit == 'y':
    reviews_df = reviews_df.sample(n=num_2_samp, replace=False, random_state=RANDOM_SEED)
    print('Sample size:', len(reviews_df))
else:
    print('Unsampled size:', len(reviews_df))

print('Script: 04.00.03 [Sampling mode settings set] completed')

# =============================================================================
# 04.01.01 | Create a list of products
# =============================================================================
if create_jlines == 'n':
    print('Script: 04.01.01 [Create jsonlines file] skipped')
else:
    headers = ['reviewerID', 'reviewText']
    if sampleit == 'y':
        out_file_name = "../data/jsonlines/collection_reviews2.jsonl"
    else:
        out_file_name = "../data/jsonlines/collection_reviews.jsonl"
    nlp_df_reviewer = gen_jlines(headers, reviews_df, out_file_name)
    print('Script: 04.01.01 [Create jsonlines file] completed')

# =============================================================================
# 04.02.01 | Readin jsonlines file
# =============================================================================
# Set up blank dictionaries to read-in jsonlines file
# Note this section is mostly necessary if you skip creating the jsonlines file in 04.01.01
# It is recommended to skip because it takes forever to process the dataframe

labels={'labels':[]}
text={'text':[]}

# Readin jsonlines file
with open(out_file_name, 'rb') as f:
    for item in json_lines.reader(f):
        labels['labels'].append(item['reviewerID'])
        text['text'].append(item['reviewText'])

# The read in creates two dataframes one for labels, one for position; this just joins them together by position
data = pd.concat([pd.DataFrame(labels),pd.DataFrame(text)], axis=1)
print(len(data))
print('Script: 04.02.01 [Readin jsonlines file] completed')

# =============================================================================
# 04.02.02 | Sample jsonlines file
# =============================================================================
# Samples the dataframe for quick active development; this will be disabled once development is done

#if sampleit == 'y':
#    data=data.sample(n=num_2_samp, replace=False, random_state=RANDOM_SEED)
#    print(len(data))
#    print('Script: 04.02.02 [NLP dataframe sample] completed')
##else:
#    print('Script: 04.02.02 [NLP dataframe sample] skipped')

# I dont think i need this, probably delete
data=data.reset_index()
print(len(data))

# =============================================================================
# 04.03.01 | Stage text for cleansing
# =============================================================================
# create empty list to store text documents
text_body = []

# for loop which appends the text to the text_body list
for i in range(0, len(data)):
    temp_text = data['text'].iloc[i]
    text_body.append(temp_text)

print('Script: 04.03.01 [Create text body] completed')

# =============================================================================
# 04.03.02 | Finally clean/process the text
# =============================================================================

# empty list to store processed documents
processed_text = []
# for loop to process the text to the processed_text list
for i in text_body:
    text = clean_docs(i)
    processed_text.append(text)

print('Script: 04.03.02 [Create text body] completed')

# Note: the processed_text is the PROCESSED list of documents read directly form
# the csv.  Note the list of words is separated by commas.

# =============================================================================
# 04.03.03 | Rebuild body of text post processing
# =============================================================================
# stitch back together individual words to reform body of text

final_processed_text = []

for i in processed_text:
    temp_DSI = ' '.join(i)
    final_processed_text.append(temp_DSI)

# Note: We stitched the processed text together so the TFIDF vectorizer can work.
# Final section of code has 3 lists used.  2 of which are used for further processing.
# (1) text_body - unused, (2) processed_text (used in W2V),
# (3) final_processed_text (used in TFIDF), and (4) DSI titles (used in TFIDF Matrix)

print('Script: 04.03.03 [Rebuilt text post processing] completed')

# =============================================================================
# 04.03.04 | Processing labels to lists
# =============================================================================

#create empty list to store labels
labels=[]

#for loop which appends the DSI title to the titles list
for i in range(0,len(data)):
    temp_text=data['labels'].iloc[i]
    labels.append(temp_text)

print('Script: 04.03.04 [Itemize labels] completed')

# =============================================================================
# 04.04.01 | Sklearn TFIDF
# =============================================================================
# note the ngram_range will allow you to include multiple words within the TFIDF matrix
# Call Tfidf Vectorizer
Tfidf = TfidfVectorizer(ngram_range=(1, 1))

# fit the vectorizer using final processed documents.  The vectorizer requires the
# stiched back together document.

TFIDF_matrix = Tfidf.fit_transform(final_processed_text)

# creating dataframe from TFIDF Matrix
matrix = pd.DataFrame(TFIDF_matrix.toarray(), columns=Tfidf.get_feature_names(), index=labels)

matrix.to_csv("../data/tfidf/tfidf_matrix.csv")

print('Script: 04.04.01 [Sklearn TFIDF, write tfidf] completed')

# =============================================================================
# 04.05.01 | K Means Clustering - TFIDF
# =============================================================================
# Set number of clusters
km = KMeans(n_clusters=k, random_state=RANDOM_SEED)
km.fit(TFIDF_matrix)
clusters = km.labels_.tolist()

terms = Tfidf.get_feature_names()
Dictionary = {'Reviewer': labels, 'Cluster': clusters, 'Text': final_processed_text}
frame = pd.DataFrame(Dictionary, columns=['Cluster', 'Reviewer', 'Text'])

frame = pd.concat([frame, data['labels']], axis=1)

frame['record'] = 1

print('Script: 04.05.01 [K Means Clustering] completed')

# =============================================================================
# 04.05.02 | Pivot table to see see how clusters compare to categories
# =============================================================================

pivot = pd.pivot_table(frame, values='record', index='labels',
                       columns='Cluster', aggfunc=np.sum)

print(pivot)

pivot.to_csv("../output/clusters/clusters_tfidf.csv")

print('Script: 04.05.02 [K Means Pivot] completed')

# =============================================================================
# 04.05.03 | Top Terms per cluster
# =============================================================================

print("Top terms per cluster:")
# sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1]

terms_dict = []

# save the terms for each cluster and document to dictionaries.  To be used later
# for plotting output.

# dictionary to store terms and titles
cluster_terms = {}
cluster_title = {}

for i in range(k):
    print("Cluster %d:" % i),
    temp_terms = []
    temp_titles = []
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind])
        terms_dict.append(terms[ind])
        temp_terms.append(terms[ind])
    cluster_terms[i] = temp_terms

    print("Cluster %d titles:" % i, end='')
    temp=frame[frame['Cluster']==i]
    for title in temp['Cluster']:
        #print(' %s,' % title, end='')
        temp_titles.append(title)
    cluster_title[i]=temp_titles

print('Script: 04.05.03 [Top terms per cluster] completed')

# =============================================================================
# 04.06.01 | TF-IDF Plotting - mds algorithm
# =============================================================================
# convert two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.

mds = MDS(n_components=2, dissimilarity="precomputed", random_state=RANDOM_SEED)
cluster_and_plot(mds, TFIDF_matrix, clusters, cluster_title, 'precomputed')
print('Script: 04.06.01 [TF-IDF Plot Plotted] completed')

# =============================================================================
# 04.06.02 | TF-IDF Plotting - mds algorithm
# =============================================================================
mds = TSNE(n_components=2, metric="euclidean", random_state=RANDOM_SEED)
cluster_and_plot(mds, TFIDF_matrix, clusters, cluster_title, 'euclidean')
print('Script: 04.06.02 [TF-IDF Plot Plotted] completed')
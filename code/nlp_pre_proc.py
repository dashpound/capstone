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
import pandas as pd
import os
import io
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
import jsonlines

# Import modules (other scripts)
from data_load import reviews_df
from environment_configuration import set_palette
from functions import clean_docs
from functions import gen_jlines

print('Script: 04.00.02 [Import Packages] completed')

# %%
# =============================================================================
# 04.01.01 | Create a list of products
# =============================================================================
df2 = reviews_df[reviews_df['reviewerID'].isin(['AO94DHGC771SJ', 'AMO214LNFCEI4'])]
headers = ['reviewerID', 'reviewText']

nlp_df_reviewer = gen_jlines(headers, df2, "../data/jsonlines/collection_reviews.jsonlines")

print('Script: 04.01.01 [Collect Text] completed')
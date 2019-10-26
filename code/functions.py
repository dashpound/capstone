# ===============================================================================
# 00.01.01 | Functions | Documentation
# ===============================================================================
# Name:               01_environment_configuration
# Author:             Julia
# Last Edited Date:   10/10/19
# Description:        Global user defined functions stored here to be called
# 
# Notes:               
# Warnings:            
# Outline:             
#                      
# =============================================================================
# 00.01.02 | Import packages
# =============================================================================
import pandas as pd
import re,string
from nltk.corpus import stopwords

# Import packages
import re, string
from nltk.corpus import stopwords

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

# =============================================================================
# 00.02.01 | Convert Pivot to Data Frame
# =============================================================================
# Convert Base Pivot to Data Frame
def conv_pivot2df(pivot_name):
    '''Pandas pivot table to data frame; requires pandas'''
    pivot_name.columns = pivot_name.columns.droplevel(0)
    pivot_name = pivot_name.reset_index().rename_axis(None, axis=1)
    pivot_name = pd.DataFrame(pivot_name)

print('Script: 00.02.01 [Convert Pivot to DF] defined')

# =============================================================================
# 00.03.01 | Define Cleanse text function
# =============================================================================
# For text preprocessing
def clean_docs(doc):
    '''Cleanse document'''
    #split document into individual words
    tokens=doc.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 4]
    tokens = [word for word in tokens if len(word) < 21]
    #lowercase all words
    tokens = [word.lower() for word in tokens]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # word stemming
    # ps=PorterStemmer()
    # tokens=[ps.stem(word) for word in tokens]
    return tokens

print('Script: 00.03.01 [Clean Docs] Defined')

# =============================================================================
# 00.03.02 | Define Collect Text Function
# =============================================================================
# Aggregate all reviews to single list for each product
def collect_text(df, groupby_column, return_column):
    # Find all unique keys in data frame
    uniquez = df[groupby_column].unique()
    by_group=[]
    label_store=[]
    # Create empty list; goal is to return a single reviewer and a concatenated reviews (two items per person)
    for i in range(0, len(uniquez)):
        """For each unique key in the dataframe, filter the dataframe to only that unique key then..."""
        f = df[df[groupby_column] == uniquez[i]]
        agg_values = []
        for j in range(0, len(f)):
            """for each row in the dataframe take the requested column and concatenate it together"""
            temp_text = f[return_column].iloc[j]
            agg_values.append(temp_text)
        """Concatenate all the items such that there is one row and one column per"""
        separator = ', '
        by_group.append(separator.join(agg_values))
        label_store.append(uniquez[i])
        a = dict(zip(label_store, by_group))
        #print('Row:', i, "completed")
    return a

print('Script: 00.03.02 [Collect Text] Defined')

# =============================================================================
# 00.03.03 | Define Generate jsonlines function
# =============================================================================

def gen_jlines(headers, dataframe, output):
    """Note as currently set up, the list of headers will be the "groupby" column and the results columns (position 0 & 1)"""
    data = collect_text(dataframe, headers[0], headers[1])
    cr_df = pd.DataFrame.from_dict(data, orient='index').reset_index()
    cr_df.columns = headers
    cr_df.to_json(output,
                  orient="records",
                  lines=True)
    return cr_df

print('Script: 00.03.03 [Generate Jsonlines] Defined')

# =============================================================================
# 00.03.04 | Cluster and Plot NLP
# =============================================================================

def cluster_and_plot(mds_alg, TFIDF_matrix, clusters, cluster_title,output):
    mds_alg
    dist = 1 - cosine_similarity(TFIDF_matrix)
    pos = mds_alg.fit_transform(dist)  # shape (n_components, n_samples)
    xs, ys = pos[:, 0], pos[:, 1]
    print('Function: 00.03.04 [TF-IDF set clustering plot] completed')

    # set up colors per clusters using a dict.  number of colors must correspond to K
    cluster_colors = {0: 'black', 1: 'orange', 2: 'blue', 3: 'rosybrown', 4: 'firebrick',
                      5: 'red', 6: 'darksalmon', 7: 'sienna'}

    # set up cluster names using a dict.
    cluster_dict = cluster_title
    print('Function: 00.03.04 [TF-IDF set colors for plot] completed')

    # create data frame that has the result of the MDS plus the cluster numbers and titles
    df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=range(0, len(clusters))))

    # group by cluster
    groups = df.groupby('label')

    fig, ax = plt.subplots(figsize=(12, 12))  # set size
    ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling
    print('Function: 00.03.04  [TF-IDF Stage the data] completed')

    # iterate through groups to layer the plot
    # note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
    for name, group in groups:
        ax.plot(group.x, group.y,
                marker='o', linestyle='', ms=12,
                label=cluster_dict[name], color=cluster_colors[name],
                mec='none')
        ax.set_aspect('auto')
        ax.tick_params( \
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=True)
        ax.tick_params( \
            axis='y',  # changes apply to the y-axis
            which='both',  # both major and minor ticks are affected
            left=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelleft=True)

    plt.title('TF-IDF Clustering | MDS Algorithm')
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))      #show legend with only 1 point
    # The following section of code is to run the k-means algorithm on the doc2vec outputs.
    # note the differences in document clusters compared to the TFIDF matrix.
    plt.savefig('../output/clusters/' + output + '.png')
    plt.show()


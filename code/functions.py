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
            #print(agg_values)
        """Concatenate all the items such that there is one row and one column per"""
        separator = ', '
        by_group.append(separator.join(agg_values))
        label_store.append(uniquez[i])
        a = dict(zip(label_store, by_group))
    return a

print('Script: 00.03.02 [Collect Text] Defined')

# =============================================================================
# 00.03.04 | Define Generate jsonlines function
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

print('Script: 00.03.04 [Generate Jsonlines] Defined')


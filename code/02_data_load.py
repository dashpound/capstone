# Name: 02_data_load
# Author: Julia
# Last Edited Date: 10/8/19

# Description:
# Loads review data and verifies there is data.
# Loads product metadata and verifies there is data.


# Notes:


# Warnings:
# The data checks are not automated right now but they are a starting point.


# Outline:
# Unpack review data.
# Verify review data loaded successfully.
# Unpack product metadata.
# Verify product metadata loaded successfully.


# =============================================================================
# Load Reviews Data
# =============================================================================
# Unpack review data where reviewer has left at least 5 reviews
with open (os.path.join(data_path, reviews_path), 'rb') as pickle_file:
    reviews_df = pickle.load(pickle_file)
    
    
# =============================================================================
# Verify Reviews Data Loaded Successfully
# =============================================================================
print('Review data shape: ', reviews_df.shape)
print('Review data columns: ', reviews_df.columns)
print(reviews_df.head())


# =============================================================================
# Load Product Metadata Data
# =============================================================================
# Performing garbage collect just in case
gc.collect()

# Unpack metadata
with open (os.path.join(data_path, metadata_path), 'rb') as pickle_file:
    metadata_df = pickle.load(pickle_file)   
    
    
# =============================================================================
# Verify Reviews Data Loaded Successfully
# =============================================================================
print('Metadata data shape: ', metadata_df.shape)
print('Metadata data columns: ', metadata_df.columns)
print(metadata_df.head())
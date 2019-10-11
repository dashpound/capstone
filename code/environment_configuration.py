# ===============================================================================
# 01.01.01 | Environment Configuration | Documentation
# ===============================================================================
# Name:               01_environment_configuration
# Author:             Julia
# Last Edited Date:   10/10/19
# Description:        Loads packages, sets working directory, 
#                     and defines global variables.
# Notes:              Must set your working directory outside of this script 
#                     to the location of the repo.
# Warnings:           The setting of the working directory is not automated.
# Outline:            Set your working directory to the code
#                     folder of the repo location (outside of this script).
# =============================================================================
# 01.01.02 | Import Packages
# =============================================================================
import seaborn as sns
import os

print('Script: 01.01.02 [Import Packages] completed')

# =============================================================================
# 01.01.03 |Set Working Directory and Paths
# =============================================================================
# set working directory and create variable

# Get original working directory
owd = os.getcwd()

# Get out of the code folder
os.chdir("..")

# Set that as the working directory variable
working_directory = os.getcwd()

# Switch all the backlashes to forward slashes so it works with "Path"
working_directory = working_directory.replace('\\', '/')

# Switch the working directory back to default
os.chdir(owd)

# other file paths
data_path = '/data'
modeling_path = '../output/models'

# define data paths
reviews_path = '/reviewsdata_5.pkl'
metadata_path = '/metadata.pkl'
qa_path = '/QA_data.pkl'

print('Script: 01.01.03 [Update Data Paths] completed')

# =============================================================================
# 01.01.04 | Define Other Global Variables
# =============================================================================
# custom color palette
my_palette = sns.color_palette() # 10 colors

# set seed for reproducability
RANDOM_SEED = 42

print('Script: 01.01.04 [Define Other Global Variables] completed')
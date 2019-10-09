# Name: 01_environment_configuration
# Author: Julia
# Last Edited Date: 10/8/19

# Description:
# Loads packages, sets working directory, and defines global variables.


# Notes:
# Must set your working directory outside of this script to the location of the repo.

# Warnings:
# The setting of the working directory is not automated.


# Outline:
# Set your working directory to the repo location (outside of this script).
# Imports needed packages.
# Defines working directory and other relevant file paths.
# Defines other global variables.


# =============================================================================
# Import Packages
# =============================================================================
# data loading
import pickle

# data manipulation/visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# misc
import gc 
import os


# =============================================================================
# Set Working Directory and Paths
# =============================================================================
# set working directory and create variable
repo_path = 'capstone\\code'
os.chdir(os.path.join(os.getcwd(), repo_path))
working_directory = os.getcwd()

# define data paths
reviews_path = 'reviewsdata_5.pkl'
metadata_path = 'metadata.pkl'

# other file paths
data_path = '../data'
modeling_path = '../output/models'


# =============================================================================
# Define Other Global Variables
# =============================================================================
# custom color palette
my_palette = sns.color_palette() # 10 colors

# set seed for reproducability
RANDOM_SEED = 42
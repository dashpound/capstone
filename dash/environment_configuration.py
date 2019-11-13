# ===============================================================================
# 01.00.01 | Environment Configuration | Documentation
# ===============================================================================
# Name:               01_environment_configuration
# Author:             Rodd
# Last Edited Date:   11/9/19
# Description:        Loads packages, sets working directory, 
#                     and defines global variables.
# Notes:              Must set your working directory outside of this script 
#                     to the location of the repo.
# Warnings:           The setting of the working directory is not automated.
# Outline:            Set your working directory to the code
#                     folder of the repo location (outside of this script).
#
#
# =============================================================================
# 01.00.02 | Import Packages
# =============================================================================
import seaborn as sns
import matplotlib.pyplot as plt
import os

print('Script: 01.00.02 [Import Packages] completed')

# =============================================================================
# 01.01.01 |Set Working Directory and Paths
# =============================================================================
# set working directory and create variable

# Get original working directory
owd = os.getcwd()

# Get out of the dash folder
os.chdir("..")

# Set that as the working directory variable
working_directory = os.getcwd()

# Switch all the backlashes to forward slashes so it works with "Path"
working_directory = working_directory.replace('\\', '/')

# Switch the working directory back to default
os.chdir(owd)

# other file paths
data_path = '/data/pickles/enhanced'
modeling_path = '../output/models'
dash_data_path = '/dash/data'

# define data paths
reviews_ind_path = '/reviews_meta_combined_individual.pkl'
reviews_agg_path = '/reviews_meta_combined_aggregated.pkl'
products_path = '/product_metadata_no_one_hot_encoding.pkl'

print('Script: 01.01.01 [Set working directory and other paths] completed')


# =============================================================================
# 01.02.01 | Define Other Global Variables
# =============================================================================

# defining length of tables
PAGE_SIZE = 11

# used for controlling filtering of tables
operators = [['ge ', '>='],
             ['le ', '<='],
             ['lt ', '<'],
             ['gt ', '>'],
             ['ne ', '!='],
             ['eq ', '='],
             ['contains '],
             ['datestartswith ']]

print('Script: 01.02.01 [Define Other Global Variables] completed')


# =============================================================================
# 01.03.01 | Define Functions
# =============================================================================
# used to help with filtering
# found this code here: https://dash.plot.ly/datatable/callbacks
def split_filter_part(filter_part):
    for operator_type in operators:
        for operator in operator_type:
            if operator in filter_part:
                name_part, value_part = filter_part.split(operator, 1)
                name = name_part[name_part.find('{') + 1: name_part.rfind('}')]

                value_part = value_part.strip()
                v0 = value_part[0]
                if (v0 == value_part[-1] and v0 in ("'", '"', '`')):
                    value = value_part[1: -1].replace('\\' + v0, v0)
                else:
                    try:
                        value = float(value_part)
                    except ValueError:
                        value = value_part

                # word operators need spaces after them in the filter string,
                # but we don't want these later
                return name, operator_type[0].strip(), value

    return [None] * 3

print('Script: 01.03.01 [Define Functions] completed')
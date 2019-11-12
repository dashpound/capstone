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

# set seed for reproducability
RANDOM_SEED = 42

print('Script: 01.02.01 [Define Other Global Variables] completed')


# =============================================================================
# 01.03.01 | Define Functions
# =============================================================================
# custom color palette
def set_palette():
    my_palette = sns.color_palette() # 10 colors
    sns.set_palette(my_palette)
    sns.palplot(my_palette)
    plt.show()
    
# combining levels of categorical variable into level of 'Other' based on threshold
threshold = 100 

def set_levels(df, col):
    df = df.copy()
    for i in col.unique():
        if len(df.loc[col == i]) < threshold:
            df.loc[col == i] = 'Other'
    return df

# define styling for plots
def plot_params():
    sns.set(font_scale=1.25)
    sns.set_style("darkgrid")
    plt.gca().xaxis.grid(True)
    plt.rcParams['font.weight'] = "bold"
    plt.rcParams['font.sans-serif'] = "Calibri"

# function for labeling bars - both vertical and horizontal
# https://stackoverflow.com/questions/43214978/seaborn-barplot-displaying-values
def show_values_on_bars(axs, h_v="v", space=0.4):
    def _show_on_single_plot(ax):
        if h_v == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height()
                value = int(p.get_height())
                ax.text(_x, _y, value, ha="center") 
        elif h_v == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height() 
                value = int(p.get_width())
                ax.text(_x, _y, value, ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)

print('Script: 01.03.01 [Define Functions] completed')
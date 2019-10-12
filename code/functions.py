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
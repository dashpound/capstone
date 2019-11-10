#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Author: Hemant Patel
Date: 11/3/2019

Instructions: Install the following using your terminal...

pip install dash==1.4.1  # The core dash backend
pip install dash-daq==0.2.1  # DAQ components (newly open-sourced!)

"""
# ===============================================================================
# 02.00.01 | Dashboard App | Documentation
# ===============================================================================
# Name:               02_app
# Author:             Rodd
# Last Edited Date:   11/9/19
# Description:        Loads packages, loads and summarizes data, and defines dash components.
#  
#                   
# Notes:              Must install dash outside of this script.
#                        pip install dash==1.4.1  # The core dash backend
#                        pip install dash-daq==0.2.1  # DAQ components (newly open-sourced!)
#                    During development, debug=True so can test changes real-time.
#                     
#
# Warnings:           Cannot filter the reviews aggregated data to Camera & Photo.
#
#
# Outline:            Import packages.
#                     Load data.
#                     Create summary data frames.
#                     Define dash layout.
#                     Define dash reactive components.
#                     Run dash.
#
#
# =============================================================================
# 02.00.02 | Import Packages
# =============================================================================
# Import packages
import pandas as pd
import pickle
from pathlib import Path
import gc

# Import modules (other scripts)
from environment_configuration import working_directory, data_path
from environment_configuration import reviews_ind_path, reviews_agg_path, products_path

# Dash packages
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import plotly.express as px
from dash.dependencies import Input, Output


# =============================================================================
# 02.01.01| Import Data
# =============================================================================
# Load individual review level dataset
with open(Path(working_directory + data_path + reviews_ind_path), 'rb') as pickle_file:
    review_data_ind = pickle.load(pickle_file)
    review_data_ind = pd.DataFrame(review_data_ind)

gc.collect()

# Load aggregeated reviewer level dataset
with open(Path(working_directory + data_path + reviews_agg_path), 'rb') as pickle_file:
    review_data_agg = pickle.load(pickle_file)
    review_data_agg = pd.DataFrame(review_data_agg)

gc.collect()

# Load product level metadata dataset
with open(Path(working_directory + data_path + products_path), 'rb')  as pickle_file:
    product_data = pickle.load(pickle_file)
    product_data = pd.DataFrame(product_data)
    
gc.collect()


# =============================================================================
# 02.02.01| Filter Data to Camera & Photo
# =============================================================================
review_data_ind = review_data_ind[review_data_ind['category2_t']=='Camera & Photo']
product_data = product_data[product_data['category2_t']=='Camera & Photo']
# THERE IS A PROBLEM TRYING TO FILTER THE AGGREGATED DATA! CAN'T DO THIS.


# =============================================================================
# 02.03.01| Define Summary Data Frames
# =============================================================================
# top 10 products
top_10_products = review_data_ind.groupby('asin').size().reset_index(name='count').sort_values('count', ascending=False).head(10)
top_10_products = pd.merge(product_data[['title','asin']],top_10_products, on='asin', how='inner')
# some of these titles are rather long. let's select the first n number of characters
top_10_products['title'] = top_10_products['title'].str[:60]

top_10_products = top_10_products.sort_values('count', ascending=True)

# number of reviewers & products
number_reviewers = review_data_ind['reviewerID'].nunique()
number_products = review_data_ind['asin'].nunique()


# =============================================================================
# 02.04.01| Dash Layout
# =============================================================================
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# colors are in spirit of Amazon color palette
colors = {'background': '#000000',
          'text': '#FF9900',
          'subtext': '#fbffae',
          'color1': '#146eb4',
          'color2': '#232f3e'}

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
        
        # CognoClick Logo
        html.Div(children=[html.Img(src=app.get_asset_url("../assets/CognoClick_upscaled_logo.jpg"),
                       id="cognoclick-logo",
                       style={'height':'35px', 
                              'width':'auto', 
                              'margin-top':'10px',
                              'margin-left':'10px'})]), 
    
#        # Amazon Logo - want these to appear side by side
#        html.Div(children=[html.Img(src=app.get_asset_url("../assets/Amazon_Logo.png"),
#                       id="amazon-logo",
#                       style={'height':'32px', 
#                              'width':'auto', 
#                              'background':'#FFFFFF',
#                              'margin-top':'10px',
#                              'margin-right':'15px'})]), 

        # Title - Don't love the spacing but this is fine for now
        html.H1(children='Amazon Recommendation Engine',
                style={'textAlign': 'center',
                       'height': '20px',
                       'margin-bottom': '40px',
                       'color': colors['text']}),
    
        # Top 10 Products Bar Chart
        dcc.Graph(
            id='top-10-graph',
            figure={
                'data': [go.Bar(y=top_10_products['title'],
                                x=top_10_products['count'], 
                                orientation='h',
                                marker_color=colors['color1'])],   
                'layout': {'title': 'Top 10 Products',
                           # titles are long so need to add a hefty left margin
                           'margin': {'l':500, 'pad':4}}}), 
                           #'margin': go.layout.Margin(l=500,pad=4)
                          
        # Creating tabs to use to add in the recommendation components
        html.Div(children=[
        	dcc.Tabs(
            	id="tabsID",
            	children=[
        		dcc.Tab(label='Product Recommendations'), 
        					 #children=[product_wise_layout]),	
        		dcc.Tab(label='User Recommendations')], 
        					 #children=[user_wise_layout]),
        		
                ## Specifies the tab to be displayed by default
            	value="Product Recommendations")])  
    ])

    
# =============================================================================
# 02.05.01| Dash Reactive Components
# =============================================================================

    
    
# =============================================================================
# 02.06.01| Run Dash
# =============================================================================                               
if __name__ == '__main__':
    # turning debug on while app is being built
    app.run_server(debug=True)
    # app.run_server(dev_tools_hot_reload=False)
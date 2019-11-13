#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Author: Hemant Patel
Date: 11/3/2019

Instructions: Install the following using your terminal...

pip install dash==1.4.1  # The core dash backend
pip install dash-daq==0.2.1  # DAQ components (newly open-sourced!)
pip install dash-bootstrap-components  # Responsive layouts and components
Visit the following location in your web browser to see app: http://127.0.0.1:8050/ 

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
from environment_configuration import working_directory, data_path, dash_data_path
from environment_configuration import reviews_ind_path, reviews_agg_path, products_path
from environment_configuration import PAGE_SIZE, operators, split_filter_part

# Dash packages
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import plotly.express as px
import dash_bootstrap_components as dbc
import dash_table as dt
from dash.dependencies import Input, Output

# Dash data table
import dash_table
import dash_html_components as html

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

# Load sample mapped product data
sample_mapped_product = pd.read_excel(Path(working_directory + dash_data_path + '/Sample_Mapped_Product_Data.xlsx'))

sample_mapped_product['Mapped Product']=sample_mapped_product['Mapped Product'].astype(str)
sample_mapped_product['Product Code']=sample_mapped_product['Product Code'].astype(str)
sample_mapped_product['Product Name'] = sample_mapped_product['Product Name'].str[:60] # only showing first 60 chars

#Load sample mapped reviewer data
sample_mapped_reviewer = pd.read_excel(Path(working_directory + dash_data_path + '/Sample_Mapped_Reviewer_Data.xlsx'))

sample_mapped_reviewer['Product Name'] = sample_mapped_reviewer['Product Name'].str[:60] # only showing first 60 chars
sample_mapped_reviewer['Product Code']=sample_mapped_reviewer['Product Code'].astype(str)


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


# =============================================================================
# 02.04.01| Dash Layout
# =============================================================================
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# colors are in spirit of Amazon color palette
colors = {#'background': '#000000',
          'background': '#FFFFFF',
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
                'layout': {'title': 'Top 10 Products Overall',
                           # titles are long so need to add a hefty left margin
                           'margin': {'l':500, 'pad':4}}}), 
                           #'margin': go.layout.Margin(l=500,pad=4)
                          
        # Creating tabs to use to add in the recommendation components
       html.Div([dcc.Tabs(id="tabs", children=[
                dcc.Tab(label='Product Recommendations', children=[
                # Product Recommendation Tab
                dash_table.DataTable(
                            id='product-table',
                            columns=[{"name": i, "id": i} for i in sample_mapped_product.columns],
                            page_current=0,
                            page_size=PAGE_SIZE,
                            page_action='custom',
                        
                            filter_action='custom',
                            filter_query='' ,
                            style_cell={'padding': '5px'},
                            style_cell_conditional=[
                                {
                                    'if': {'column_id': c},
                                    'textAlign': 'left'
                                } for c in ['Product Name']
                            ],
                            style_header={
                                'backgroundColor': 'white',
                                'fontWeight': 'bold'
                            },
                            style_data_conditional=[
                                {
                                    'if': {'row_index': 'odd'},
                                    'backgroundColor': 'rgb(248, 248, 248)'
                                }
                            ],        
                        ),
                dcc.Markdown('''
                             ###### Each column can be filtered based on user input.
                             
                             ###### For string columns, just enter a partial string such as "Nook."
                             
                             ###### Exception: For product columns, use quotes around filter, such as "328."
                             
                             ###### For numeric columns, filters such as "=5" or ">=200" are valid filters.
                             
                             ###### Use "Enter" to initiate and remove filters.
                             
                             ''')  ]),
    
          # User Recommendation Tab
          dcc.Tab(label='User Recommendations', children=[
                
                # User Table
                dash_table.DataTable(
                        id='user-table',
                        columns=[{"name": i, "id": i} for i in sample_mapped_reviewer.columns],
                        page_current=0,
                        page_size=PAGE_SIZE,
                        page_action='custom',
                    
                        filter_action='custom',
                        filter_query='' ,
                        style_cell={'padding': '5px'},
                        style_cell_conditional=[
                            {
                                'if': {'column_id': c},
                                'textAlign': 'left'
                            } for c in ['Product Name']
                        ],
                        style_header={
                            'backgroundColor': 'white',
                            'fontWeight': 'bold'
                        },
                        style_data_conditional=[
                            {
                                'if': {'row_index': 'odd'},
                                'backgroundColor': 'rgb(248, 248, 248)'
                            }
                        ]),
                
                dcc.Markdown('''
                             ###### Each column can be filtered based on user input.
                             
                             ###### For string columns, just enter a partial string such as "Nook."
                             
                             ###### Exception: For product columns, use quotes around filter, such as "328."
                             
                             ###### For numeric columns, filters such as "=5" or ">=200" are valid filters.
                             
                             ###### Use "Enter" to initiate and remove filters.
                             
                             ''')])
        ])])
    
  ])
    
# =============================================================================
# 02.05.01| Dash Reactive Components | Product Table
# =============================================================================
@app.callback(
    Output('product-table', "data"),
    [Input('product-table', "page_current"),
     Input('product-table', "page_size"),
     Input('product-table', "filter_query")])


def update_table(page_current,page_size, filter):
    print(filter)
    filtering_expressions = filter.split(' && ')
    dff = sample_mapped_product # WILL NEED TO CHANGE THIS
    for filter_part in filtering_expressions:
        col_name, operator, filter_value = split_filter_part(filter_part)

        if operator in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
            # these operators match pandas series operator method names
            dff = dff.loc[getattr(dff[col_name], operator)(filter_value)]
        elif operator == 'contains':
            dff = dff.loc[dff[col_name].str.contains(filter_value)]
        elif operator == 'datestartswith':
            # this is a simplification of the front-end filtering logic,
            # only works with complete fields in standard format
            dff = dff.loc[dff[col_name].str.startswith(filter_value)]

    return dff.iloc[
        page_current*page_size:(page_current+ 1)*page_size
    ].to_dict('records')


# =============================================================================
# 02.05.02| Dash Reactive Components | User Table
# =============================================================================
@app.callback(
    Output('user-table', "data"),
    [Input('user-table', "page_current"),
     Input('user-table', "page_size"),
     Input('user-table', "filter_query")])

def update_table2(page_current,page_size, filter):
    print(filter)
    filtering_expressions = filter.split(' && ')
    dff = sample_mapped_reviewer # WILL NEED TO CHANGE THIS
    for filter_part in filtering_expressions:
        col_name, operator, filter_value = split_filter_part(filter_part)

        if operator in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
            # these operators match pandas series operator method names
            dff = dff.loc[getattr(dff[col_name], operator)(filter_value)]
        elif operator == 'contains':
            dff = dff.loc[dff[col_name].str.contains(filter_value)]
        elif operator == 'datestartswith':
            # this is a simplification of the front-end filtering logic,
            # only works with complete fields in standard format
            dff = dff.loc[dff[col_name].str.startswith(filter_value)]

    return dff.iloc[
        page_current*page_size:(page_current+ 1)*page_size
    ].to_dict('records')
    
    
# =============================================================================
# 02.06.01| Run Dash
# =============================================================================                               
if __name__ == '__main__':
    # turning debug on while app is being built
    app.run_server(debug=True)
    # app.run_server(dev_tools_hot_reload=False)
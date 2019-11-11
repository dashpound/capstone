#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Author: Hemant Patel
Date: 11/3/2019

Instructions: Install the following using your terminal...

pip install dash==1.4.1  # The core dash backend
pip install dash-daq==0.2.1  # DAQ components
pip install dash-bootstrap-components  # Responsive layouts and components
Visit the following location in your web browser to see app: http://127.0.0.1:8050/ 

"""


# Import packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches
import pickle


# Load individual review level dataset
with open ('/Users/Hemant/Desktop/Amazon Review Data/reviews_meta_combined_individual.pkl', 'rb') as pickle_file:
    review_data_ind = pickle.load(pickle_file)


# Load aggregeated reviewer level dataset
with open ('/Users/Hemant/Desktop/Amazon Review Data/reviews_meta_combined_aggregated.pkl', 'rb') as pickle_file:
    review_data_agg = pickle.load(pickle_file)


# Load product level metadata dataset
with open ('/Users/Hemant/Desktop/Amazon Review Data/product_metadata_no_one_hot_encoding.pkl', 'rb') as pickle_file:
    meta_data = pickle.load(pickle_file)


# Load sample mapped product data
sample_mapped_product = pd.read_excel('/Users/Hemant/Desktop/Sample_Mapped_Product_Data.xlsx')


#Load sample mapped reviewer data
sample_mapped_reviewer = pd.read_excel('/Users/Hemant/Desktop/Sample_Mapped_Reviewer_Data.xlsx')


# Aggregate reviews/sales by year and month
sales_YM = review_data_ind[['reviewDate']]
sales_YM['Sales_Year_Month'] = sales_YM['reviewDate'].dt.to_period('M')
sales_YM_summary = sales_YM.groupby('Sales_Year_Month').count()
sales_YM_summary = sales_YM_summary.reset_index()
sales_YM_summary = sales_YM_summary.rename(columns={'Sales_Year_Month':'Sales Period', 'reviewDate':'Sales Count'})
sales_YM_summary['Sales Period'] = sales_YM_summary['Sales Period'].astype(str)


# Determine top 10 by reviews/sales
metadata_sales = meta_data.sort_values('numberReviews', ascending=False)
metadata_sales_summary = metadata_sales[['asin', 'title', 'numberReviews', 'price_t', 'meanStarRating']].head(n=10)
metadata_sales_summary['meanStarRating'] = round(metadata_sales_summary['meanStarRating'], 1)
metadata_sales_summary = metadata_sales_summary.rename(columns={'asin':'Product Code', 'title':'Product Name', 'numberReviews':'Review Count', 'price_t':'Price', 'meanStarRating':'Average Rating'})


# Dash packages
import dash
import base64
import dash_core_components as dcc
import dash_html_components as html
import plotly.figure_factory as ff
import plotly.graph_objs as go
import plotly.express as px
import dash_bootstrap_components as dbc
import dash_table as dt
from dash.dependencies import Input, Output


# Encode CognoClick logo
cognoclick_logo = '/Users/Hemant/Desktop/Amazon Review Data/CognoClick_upscaled_logo_Adjusted.png'
cognoclick_logo_base64 = base64.b64encode(open(cognoclick_logo, 'rb').read()).decode('ascii')


# Dash layout
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

colors = {'background': '#000811',
          'text': '#aed2ff',
          'subtext': '#fbffae',
          'color1': '#ffb3ae',
          'color2': '#b3aeff',
          }

app.layout = html.Div(
        style={'backgroundColor': colors['background']}, 
        
        children=[

# Team Logo - Image        
#        html.Img(src='data:image/png;base64,{}'.format(cognoclick_logo_base64),
#                 style={'textAlign': 'center'}),        

# Team Logo - Typed
        html.H1(children='CognoClick',
                style={'textAlign': 'center',
                       'color': colors['text']}),

# Amazon KPI Section
        html.Div(children='Amazon KPIs',
                 style={'textAlign': 'center',
                        'color': colors['subtext']}),

# Amazon Monthly Sales & Top Sales
        html.Div([dbc.Row([
                  dbc.Col(html.Div([
                          dcc.Graph(id='monthly-sales',
                                     figure={'data': [
                                             {'x': sales_YM_summary['Sales Period'],
                                              'y': sales_YM_summary['Sales Count'],
                                              'name': 'Monthly Sales',
                                              'type': 'bar',
                                              'color': colors['color1']}],
                                             'layout': {'title': 'Monthly Sales Activity'}})])),
                  dbc.Col(html.Div([
                          dt.DataTable(id='top-sales-datatable',
                                        data=metadata_sales_summary.to_dict('records'),
                                        columns=[{"name": i, "id": i} for i in metadata_sales_summary.columns],
                                        style_table={'overflowX': 'scroll'},
                                        style_cell={'height': 'auto', 
                                                    'minWidth': '0px',
                                                    'maxWidth': '60px',
                                                    'whiteSpace': 'normal',
                                                    'font-family':'Arial',
                                                    'fontSize':11},
                                        style_cell_conditional=[{'if': {'column_id': c},
                                                                 'textAlign': 'left'}
                                                                 for c in ['title']],  
                                        style_data_conditional=[{'if': {'row_index': 'odd'},
                                                                 'backgroundColor': 'rgb(225, 225, 225)'}],
                                        style_header={'backgroundColor': 'rgb(145, 145, 145)', 
                                                      'fontWeight': 'bold'})]))])]),

# Average Reviewer Rating and Average Product Rating    
        html.Div([dbc.Row([
                  dbc.Col(html.Div([
                          dcc.Graph(id='reviewer-rating-histogram',
                                     figure={'data': [
                                             {'x': review_data_agg['AverageRating'],
                                              'name': 'Reviewer Average Rating',
                                              'type': 'histogram',
                                              'color': colors['color1']}],
                                             'layout': {'title': 'Average Rating Histogram by Reviewer'}})])),
                  dbc.Col(html.Div([
                          dcc.Graph(id='product-rating-histogram',
                                     figure={'data': [
                                             {'x': meta_data['meanStarRating'],
                                              'name': 'Product Average Rating',
                                              'type': 'histogram',
                                              'color': colors['color2']}],
                                             'layout': {'title': 'Average Rating Histogram by Product'}})]))])]), 

# Average Reviewer Price and Average Product Price
        html.Div([dbc.Row([
                  dbc.Col(html.Div([
                          dcc.Graph(id='reviewer-price-histogram',
                                     figure={'data': [
                                             {'x': review_data_agg['AveragePrice'],
                                              'name': 'Reviewer Average Price',
                                              'type': 'histogram',
                                              'color': colors['color1']}],
                                             'layout': {'title': 'Average Price Histogram by Reviewer'}})])),
                  dbc.Col(html.Div([
                          dcc.Graph(id='product-price-histogram',
                                     figure={'data': [
                                             {'x': meta_data['price_t'],
                                              'name': 'Product Average Price',
                                              'type': 'histogram',
                                              'color': colors['color2']}],
                                             'layout': {'title': 'Average Price Histogram by Product'}})]))])]),

# Product Recommendation Section
        html.Div(children='Product Recommendation Engine',
                 style={'textAlign': 'center',
                        'color': colors['subtext']}),

# Product Dropdown Menu
        html.Div([dcc.Dropdown(id='product-dropdown',
                     options=[{'label': i, 'value': i} for i in sample_mapped_product['Mapped Product'].unique()],
                     value='Mapped Product Code',
                     style={'width': '50%',
                            'display': 'inline-block'})]),

# Product Data Table                     
        html.Div([dt.DataTable(id='product-datatable',
                     data=sample_mapped_product.to_dict('records'),
                     columns=[{'name': i, 'id': i} for i in sample_mapped_product.columns],
                     style_table={'overflowX': 'scroll'},
                     style_cell={'height': 'auto', 
                                 'minWidth': '0px', 
                                 'maxWidth': '180px', 
                                 'whiteSpace': 'normal',
                                 'font-family':'Arial',
                                 'fontSize':11},
                     style_cell_conditional=[{'if': {'column_id': c}, 
                                              'textAlign': 'left'} 
                                              for c in ['Product Name', 'Item URL']],  
                     style_data_conditional=[{'if': {'row_index': 'odd'}, 
                                              'backgroundColor': 'rgb(225, 225, 225)'}],
                     style_header={'backgroundColor': 'rgb(145, 145, 145)', 
                                   'fontWeight': 'bold'})]),

## Dynamic Callback Filtering For Product --> CAN'T FIGURE OUT WHAT THIS IS NOT WORKING
#@app.callback(dash.dependencies.Output('product-datatable', 'data'), 
#              [dash.dependencies.Input('product-dropdown', 'value')])
#
#def update_rows(selected_value):
#    dff = sample_mapped_product[sample_mapped_product['Mapped Product'] == value]
#    return dff.to_dict('records')


# Reviewer Recommendation Section
        html.Div(children='Reviewer Recommendation Engine',
                 style={'textAlign': 'center',
                        'color': colors['subtext']}),

# Reviewer Dropdown Menu
        html.Div([dcc.Dropdown(id='reviewer-dropdown',
                     options=[{'label': i, 'value': i} for i in sample_mapped_reviewer['Mapped Reviewer'].unique()],
                     value='Mapped Reviewer ID',
                     style={'width': '50%',
                            'display': 'inline-block'})]),

# Reviewer Data Table
        html.Div([dt.DataTable(id='reviewer-datatable',
                     data=sample_mapped_reviewer.to_dict('records'),
                     columns=[{"name": i, "id": i} for i in sample_mapped_reviewer.columns],
                     style_table={'overflowX': 'scroll'},
                     style_cell={'height': 'auto', 
                                 'minWidth': '0px', 
                                 'maxWidth': '180px', 
                                 'whiteSpace': 'normal',
                                 'font-family':'Arial',
                                 'fontSize':11},
                     style_cell_conditional=[{'if': {'column_id': c}, 
                                              'textAlign': 'left'} 
                                              for c in ['Product Name', 'Item URL']],  
                     style_data_conditional=[{'if': {'row_index': 'odd'}, 
                                              'backgroundColor': 'rgb(225, 225, 225)'}],
                     style_header={'backgroundColor': 'rgb(145, 145, 145)', 
                                   'fontWeight': 'bold'})]),                     
               
    ])

# Final Step                           
if __name__ == '__main__':
    app.run_server(dev_tools_hot_reload=False)


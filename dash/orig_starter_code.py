#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Author: Hemant Patel
Date: 11/3/2019

Instructions: Install the following using your terminal...

pip install dash==1.4.1  # The core dash backend
pip install dash-daq==0.2.1  # DAQ components (newly open-sourced!)

"""

# Import modules (other scripts)
from environment_configuration import working_directory, data_path
from environment_configuration import reviews_ind_path, reviews_agg_path, products_path


# Load individual review level dataset
with open(Path(working_directory + data_path + reviews_ind_path), 'rb') as pickle_file:
    review_data_ind = pickle.load(pickle_file)


# Load aggregeated reviewer level dataset
with open(Path(working_directory + data_path + reviews_agg_path), 'rb') as pickle_file:
    review_data_agg = pickle.load(pickle_file)


# Load product level metadata dataset
with open(Path(working_directory + data_path + products_path), 'rb')  as pickle_file:
    meta_data = pickle.load(pickle_file)


# Dash layout
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import plotly.express as px

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {'background': '#000811',
          'text': '#aed2ff',
          'subtext': '#fbffae',
          'color1': '#ffb3ae',
          'color2': '#b3aeff',
          }

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
        
        html.H1(children='CognoClick',
                style={'textAlign': 'center',
                       'color': colors['text']}),

        html.Div(children='Amazon Recommendation Engine',
                 style={'textAlign': 'center',
                        'color': colors['subtext']}),

        dcc.Graph(id='histograms',
                  figure={'data': [
                          {'x': review_data_agg['AverageRating'],
                           'name': 'Reviewer Average Rating',
                           'type': 'histogram'},
                          {'x': meta_data['meanStarRating'],
                           'name': 'Product Average Rating',
                           'type': 'histogram'}],
                  'layout': {'title': 'Average Rating Histogram'}}),
    
        dcc.Graph(id='histogram-1',
                  figure={'data': [
                          {'x': review_data_agg['AveragePrice'],
                           'name': 'Reviewer Average Price',
                           'type': 'histogram',
                           'color': colors['color1']}],
                  'layout': {'title': 'Reviewer Average Price Histogram'}}),    
    
        dcc.Graph(id='histogram-2',
                  figure={'data': [
                          {'x': meta_data['price_t'],
                           'name': 'Product Average Price',
                           'type': 'histogram',
                           'color': colors['color2']}],
                  'layout': {'title': 'Product Average Price Histogram'}}),   
    ])

                                
if __name__ == '__main__':
    app.run_server(dev_tools_hot_reload=False)



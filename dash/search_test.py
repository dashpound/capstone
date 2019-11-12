# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 19:54:17 2019

@author: julia
WIP!!!!!!!!
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

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


# Dash packages
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import plotly.express as px
import dash_bootstrap_components as dbc
import dash_table as dt
from dash.dependencies import Input, Output


# =============================================================================
# 02.01.01| Import Data
# =============================================================================
# Load sample mapped product data
sample_mapped_product = pd.read_excel(Path(working_directory + dash_data_path + '/Sample_Mapped_Product_Data.xlsx'))


#Load sample mapped reviewer data
sample_mapped_reviewer = pd.read_excel(Path(working_directory + dash_data_path + '/Sample_Mapped_Reviewer_Data.xlsx'))


# =============================================================================
# 02.02.01| Dash Layout
# =============================================================================
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# colors are in spirit of Amazon color palette
colors = {'background': '#000000',
          'text': '#FF9900',
          'subtext': '#fbffae',
          'color1': '#146eb4',
          'color2': '#232f3e'}

app.layout = html.Div([ children=[
             # Product Dropdown Menu
             html.Div([dcc.Dropdown(id='product-dropdown',
                     options=[{'label': i, 'value': i} for i in sample_mapped_product['Mapped Product'].unique()],
                     value='Mapped Product Code',
                     style={'width': '50%',
                            'display': 'inline-block'})]),
              # Product Table
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
    
    html.Div(id='tabs-content')
  ])
    
    
# =============================================================================
# 02.03.01| Dash Reactive Components
# =============================================================================
#@app.callback(Output('tabs-content', 'children'),
#              [Input('tabs', 'value')])
#
#def render_content(tab):
#    if tab == 'Product Recommendations':
#        return html.Div([
#            html.H3('Product Recommendations')
#        ])
#    elif tab == 'User Recommendations':
#        return html.Div([
#            html.H3('User Recommendations')
#        ])
#    
# =============================================================================
# 02.04.01| Run Dash
# =============================================================================                               
if __name__ == '__main__':
    # turning debug on while app is being built
    app.run_server(debug=True)
    # app.run_server(dev_tools_hot_reload=False)
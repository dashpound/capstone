# ===============================================================================
# 02.00.01 | Dashboard App | Documentation
# ===============================================================================
# Name:               02_app
# Author:             Rodd/Patel
# Last Edited Date:   11/17/19
# Description:        Loads packages, loads and summarizes data, and defines dash components.
#  
#                   
# Notes:              Must install dash outside of this script.
#                        pip install dash==1.4.1  # The core dash backend
#                        pip install dash-daq==0.2.1  # DAQ components (newly open-sourced!)
#                    Dash code is finnicky on formatting and placement. There is some code that could be made into a function but dash does not like calling a function.
#                     
#
# Warnings:           Cannot filter the reviews aggregated data to Camera & Photo.
#
#
# Outline:            Import packages.
#                     Load data.
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
from math import trunc

# Import modules (other scripts)
from environment_configuration import working_directory, data_path, dash_data_path
from environment_configuration import reviews_ind_path, reviews_agg_path, products_path
from environment_configuration import colors, PAGE_SIZE, operators, split_filter_part

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
import dash_table_experiments

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
# 02.02.01| Define Summary Data Frames
# =============================================================================
# top 10 products
top_10_products = product_data.sort_values('numberReviews', ascending=False).head(10)[['title','numberReviews','price_t','category2_t','category3_t']]
top_10_products = top_10_products.sort_values('numberReviews', ascending=True)

# some of these titles are rather long. let's select the first n number of characters
top_10_products['title'] = top_10_products['title'].str[:60]


# =============================================================================
# 02.03.01| Dash Layout
# =============================================================================
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div(style={'backgroundColor': colors['d_blue_col']}, children=[
        
        # CognoClick Logo
        html.Div([dbc.Row([
                dbc.Col(html.Div(children=[html.Img(src=app.get_asset_url("../assets/CognoClick_upscaled_logo.jpg"),
                                                    id="cognoclick-logo",
                                                    style={'height':'35px', 
                                                           'width':'auto', 
                                                           'margin-top':'10px',
                                                           'margin-left':'10px'})])), 

        # Title
                dbc.Col(html.H1(children='Amazon Recommendation Engine',
                                style={'textAlign': 'center',
                                       'font-family':'Arial',
#                                       'height': '20px',
#                                       'margin-bottom': '40px',
                                       'color': colors['gray_col']})),
    
        # Amazon Logo
                dbc.Col(html.Div(children=[html.Img(src=app.get_asset_url("../assets/Amazon_Logo.png"),
                                                    id="amazon-logo",
                                                    style={'background':colors['white_col'],
                                                           'height':'32px', 
                                                           'width':'auto', 
                                                           'margin-top':'10px',
                                                           'display': 'inline-block',
                                                           'margin-left':'290px',
                                                           'margin-right': '10px'})]))])]),
    
        # Top 10 Products Bar Chart
        dcc.Graph(
            id='top-10-graph',
            figure={
                'data': [go.Bar(y=top_10_products['title'],
                                x=top_10_products['numberReviews'], 
                                orientation='h',
                                marker_color=colors['orange_col'],
                                # adding custom hover info to bar plot
                                # had to search high and low to discover that this formatting works properly
                                # <br> is used to create new lines
                                text=['<b>Number of Reviews: </b>'+'{}'.format(trunc(numberReviews))+ # need this to be integer format
                                      '<br><b>Price: </b>'+'${:.2f}'.format(price_t)+
                                      '<br><b>Category 2: </b>'+'{}'.format(category2_t)+
                                      '<br><b>Category 3: </b>'+'{}'.format(category3_t)
                                      for numberReviews, price_t, category2_t, category3_t in 
                                               zip(list(top_10_products['numberReviews']),
                                               list(top_10_products['price_t']), 
                                               list(top_10_products['category2_t']),
                                               list(top_10_products['category3_t']))],
                                hoverinfo="text",
                                hoverlabel_align = 'left'
                                )],
                'layout': {'title': 'Top 10 Products Overall',
                           'plot_bgcolor': colors['white_col'],
                           'paper_bgcolor': colors['white_col'],
                           'font': {'color': colors['black_col']},
                           # titles are long so need to add a hefty left margin
                           'margin': {'l':500, 'pad':4}}}),
                          
        # Creating tabs to use to add in the recommendation components
        html.Div([dcc.Tabs(id="tabs", 
                           colors={'border': colors['black_col'],
                                   'primary': colors['orange_col'],
                                   'background': "cornsilk"},
                           children=[
                # Product Recommendation Tab
                dcc.Tab(label='Product Recommendations', children=[
                dash_table.DataTable(
                        id='product-table',
                        columns=[{"name": i, "id": i} for i in sample_mapped_product.columns],
                        page_current=0,
                        page_size=PAGE_SIZE,
                        page_action='custom',
                        filter_action='custom',
                        filter_query='' ,
                        style_cell={'padding':'5px',
                                    'fontSize':11,
                                    'textAlign': 'left'},
                        style_header={'backgroundColor': colors['d_blue_col'],
                                      'color': colors['white_col'],
                                      'fontSize':13,
                                      'fontWeight': 'bold'},
                        style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': colors['lgray_col']}
        ]),
                # To add a new line, just add two spaces at the end of a sentence.
                # Cheating to get rid of dark background color at the end of text by adding a pad.
                dcc.Markdown('''
                             ###### Directions
                             Each column can be filtered based on user input.  
                             For string columns, just enter a partial string such as "Nook."  
                             Exception: For product columns, use quotes around filter, such as "328."  
                             For numeric columns, filters such as "=5" or ">=200" are valid filters.  
                             Use "Enter" to initiate and remove filters.  ''',
                             style={'backgroundColor': colors['white_col'],
                                    'fontSize':11,
                                    'padding':'10px'})]),
    
          # User Recommendation Tab
                dcc.Tab(label='User Recommendations', children=[
                dash_table.DataTable(
                        id='user-table',
                        columns=[{"name": i, "id": i} for i in sample_mapped_reviewer.columns],
                        page_current=0,
                        page_size=PAGE_SIZE,
                        page_action='custom',
                        filter_action='custom',
                        filter_query='' ,
                        style_cell={'padding':'5px',
                                    'fontSize':11,
                                    'textAlign': 'left'},
                        style_header={'backgroundColor': colors['d_blue_col'],
                                      'color': colors['white_col'],
                                      'fontSize':13,
                                      'fontWeight': 'bold'},
                        style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': colors['lgray_col']}]),
                # To add a new line, just add two spaces at the end of a sentence.
                # Cheating to get rid of dark background color at the end of text by adding a pad.
                dcc.Markdown('''
                             ###### Directions
                             Each column can be filtered based on user input.  
                             For string columns, just enter a partial string such as "Nook."  
                             Exception: For product columns, use quotes around filter, such as "328."  
                             For numeric columns, filters such as "=5" or ">=200" are valid filters.  
                             Use "Enter" to initiate and remove filters.  ''',
                             style={'backgroundColor': colors['white_col'],
                                    'fontSize':11,
                                    'padding':'10px'})]),
        ])])])
    
# =============================================================================
# 02.04.01| Dash Reactive Components | Product Table
# =============================================================================
@app.callback(
    Output('product-table', "data"),
    [Input('product-table', "page_current"),
     Input('product-table', "page_size"),
     Input('product-table', "filter_query")])

# tried to move this to config file and call function
# but dash does not like a function call here and requires a function definition
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
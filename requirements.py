from dash import Dash, dcc, callback, Input, Output
from dash import html
from dash import dash_table
from dash.dash_table.Format import Format, Scheme, Symbol, Group
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objs as go
from plotly.graph_objs import *
from plotly.subplots import make_subplots
import dash_mantine_components as dmc

# color and image related packages
import matplotlib
import colormaps as cmaps
import plotly.io as pio
from PIL import Image
pio.renderers.default = "plotly_mimetype"

import dbs_Dashboard as dbs
from sklearn.linear_model import LinearRegression
from scipy.interpolate import splprep, splev
import folium
from folium import plugins
import branca
import numpy as np
import pandas as pd
import re
from glob import glob
import os

# =======================================================================================
#!!! TODO
# adjust layout - fontsize
# REMOVE PREDEFINED PARAMETERS _ SHALL BE MADE INTERACTIVE
var_corr = 'pearson'
lbl_plankton = 'Picoplankton'

# =======================================================================================
# GLOABL PARAMETERS 
# general display settings
pd.set_option("display.precision", 3)

# colors
colors = {'background': 'rgba(255, 255, 255, 1)',
          'bgd_transparent': "rgba(0, 0, 0, 0)",
          'spine': 'rgba(0,0,0,1)',
          'text': 'rgba(0,67,90,255)',
          'font': "#6a6e71",
          'linecolor': "#909497", 
          'marker': 'rgba(0,0,0,1)',
          'fill_color': 'rgba(68, 68, 68, 0.3)',
          'divMap': cmaps.iceburn,
          }
transparency = 0.2
fs_text = 13
fs_label = 11
fs_label2 = 12
fs_tick = 9

# figure layout and style sheet template
layout = Layout(paper_bgcolor=colors['bgd_transparent'], plot_bgcolor=colors['background'])
external_stylesheets = [dbc.themes.SANDSTONE] 

# directories and files
path_work = os.path.dirname(os.path.realpath(__file__))
path_save = path_work + '/results/'
# creatae export folder if it does not exist
if not os.path.exists(path_save):
    os.makedirs(path_save)

# path to pre-plotted sampling location map
url = 'http://docs.google.com/spreadsheets/d/16iV9kBi2QfVCweu_LPhWxnvCdDe4av_6/edit?usp=sharing&ouid=113974104684169264313&rtpof=true&sd=true'

# !!! TODO: upload data to github
# specify how to identify pearson and spearman correlation images from result folder
file_corr_pattern =  '*Correlation_absAbund*.png'
# non-equidistant depth profile to demonstrate for environmental parameter - plankton correlation at a certain sample depth
file = path_work + '/input/correlation_mindmap.xlsx'

# specify center for geospatial map
map_center = (73.986239, -18.984375)
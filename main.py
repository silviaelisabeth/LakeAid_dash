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

import matplotlib
from matplotlib import colors 
import colormaps as cmaps
import plotly.io as pio
from PIL import Image
pio.renderers.default = "plotly_mimetype"

from sklearn.linear_model import LinearRegression
from scipy.interpolate import splprep, splev
import numpy as np
import pandas as pd
from glob import glob

import dbs_Dashboard as dbs
import requirements as req
import load_Data as ld
import dash_figures as dsf

pd.set_option("display.precision", 3)


# =======================================================================================
# Initialize the app - incorporate a template
app = Dash(__name__,  use_pages=False,  external_stylesheets=req.external_stylesheets)
app.scripts.config.serve_locally = True

# ...........................................................
# App layout
app.layout = dbc.Container([
    # TITIEL
    dbc.Row([html.Div(ld.title, className="text-primary text-left fs-2")]),
    dmc.Space(h=60),
    
    # SAMPLING LOCATION
    dbc.Row([dbc.Col([dcc.Graph(figure=dsf.fig_sampLoc, id='Trend_samplingLoc')], width=4),
             dbc.Col([html.Iframe(id='Map_samplingLoc', srcDoc=open('foliumMap_LakeAid_Greenland.html', 'r').read(),
                                  width='99%', height=450)], width=8),]),
    dmc.Space(h=50),
    dmc.Divider(),
    dmc.Space(h=50),

    # DEPTH PROFILES
    # !!! TODO: include all parameters
    dbc.Row([html.Div(ld.heading1, className="text-primary text-left fs-5")]),
    dbc.Row([dbc.Col([dbc.RadioItems(options=[{"label": x, "value": x} for x in ld.ls_enviPara], value='Oxygen', 
                                     id="DetphProf-selection", inline=False, style={"fontSize":req.fs_text},)]),
            dbc.Col([dcc.Graph(id='Depth_Profile')], width=5),
            dbc.Col([dcc.Graph(id='Depth_Profile_v2')], width=5),]),
    dmc.Space(h=10),
    dmc.Divider(),
    dmc.Space(h=50),

    # Correlation Plot/Table
    dbc.Row([html.Div(ld.heading2, className="text-primary text-left fs-5")]),
    dbc.Row([dbc.Col([dbc.RadioItems(options=[{"label": x, "value": x} for x in ld.ls_corrMeth], value='Pearson', 
                                     id="CorrMatrix-selection", inline=True, style={"fontSize":req.fs_text})]),
             html.P(id="explanation_corr_Method", style={"fontSize":req.fs_text}),]),
    dbc.Row([dbc.Col([dcc.Graph(id='Correlation-HeatMap-image')], width=5),
             dbc.Col([dash_table.DataTable(id='correlationMatrix', 
                                           columns = [{"id": i, "name": i, "type": "numeric", "format": Format().precision(3)} 
                                                      for i in ld.dic_corr['pearson'].reset_index().columns], 
                                           data=[],
                                           style_data={'whiteSpace': 'normal','height': 'auto'},
                                           fill_width=False, page_size=13, style_as_list_view=True,
                                           style_table={'overflowX':'auto', 'width':'710px', 'height':'450px', 'overflowY':'auto'},
                                           style_cell={'fontSize':req.fs_text, 'font-family':'Econ Sans', 
                                                       'color':req.colors['font']}, 
                                           style_header={'backgroundColor':req.colors['linecolor'], 'color':'white'},)],
                     width=7),]),
    dmc.Space(h=30),
    dmc.Divider(),
    dmc.Space(h=30),
    
     # Correlation Profiles Phytoplankton
    dbc.Row([html.Div(ld.heading3, className="text-primary text-left fs-5")]),
    dbc.Row([dbc.Col([dbc.RadioItems(options=[{"label": x, "value": x} for x in ld.ls_phyto], value='Picoplankton', 
                                     inline=False, style={"fontSize":req.fs_text}, id="CorrPhyto-selection")]),
            dbc.Col([dcc.Graph(id='corr_samplingDepth')], width=3),
            dbc.Col([dcc.Graph(id='corr_samplingDepth_Detail')], width=7),
             ]),
    dmc.Space(h=80),
    dbc.Row([html.P(ld.impressum, style={"fontSize":req.fs_label, "textAlign":'center'},)]),
    dmc.Space(h=20),
], fluid=True)


# .................................................................
@app.callback(Output("correlationMatrix", "data"), Input("CorrMatrix-selection", 'value'))
def updateTable(var_corr):
    if 'pearman' in var_corr:
        var_corr = 'spearman'
    elif 'earson' in var_corr:
        var_corr = 'pearson'
    return ld.dic_corr[var_corr].reset_index().to_dict('records')

# callback · Graph adjustment - Depth Profile
@app.callback(Output('Depth_Profile', 'figure'), Input('DetphProf-selection', 'value'))
def update_graph(para):
    return dsf.depth_profile(para)

# callback · Graph adjustment - Depth Profile_AVERAGE
@app.callback(Output('Depth_Profile_v2', 'figure'), Input('DetphProf-selection', 'value'))
def update_graph_AV(para):
    return dsf.depth_profileDetail(para)

# callback · adjust correlation method description
@app.callback(Output('explanation_corr_Method', "children"), Input("CorrMatrix-selection", 'value'))
def update_tex_correlation(var_corr):
    if var_corr == 'spearman':
        var_corr = 'Spearman'
    elif var_corr == 'pearson':
        var_corr = 'Pearson'
    text = ld.dic_descrip[var_corr]
    return html.P(children=text)
    
# callback · Graph adjustment - Depth Profile · AVERAGE
@app.callback(Output('Correlation-HeatMap-image', 'figure'), Input("CorrMatrix-selection", 'value'))
def update_graph_correlation(var_corr):
    if var_corr == 'Spearman':
        fig = dsf.dic_fig_corr['spearman']
    else:
        fig = dsf.dic_fig_corr['pearson']
    return fig

# callback · Graph adjustment - correlation FACTOR per sampling depth
@app.callback(Output('corr_samplingDepth', 'figure'), Input("CorrPhyto-selection", 'value'))
def update_graph_depthCorr(lbl_plankton):
    return dsf.depthCorrelation(lbl_plankton)

# callback · Graph adjustment - correlation FACTOR per sampling depth
@app.callback(Output('corr_samplingDepth_Detail', 'figure'), Input("CorrPhyto-selection", 'value'))
def update_graph_depthCorrDetail(lbl_plankton):
    return dsf.depthCorrelationDetail(lbl_plankton)


# ..................................................................................................................
# create the dashboard 
if __name__ == "__main__":
    # IP address in home wifi · 192.168.0.181
    app.run_server(debug=True)
    #app.run_server(host="0.0.0.0", port=8888)
 

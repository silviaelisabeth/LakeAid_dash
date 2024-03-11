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
import dash_figures as df


# .................................................................................................................................
# HARD CODED / STABLE FIGURES
# .................................................................................................................................
# FIGURE 1 · get specifics for sampling location where plankton was found and sort from in>out of the fjord
fig_sampLoc = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02)
fig_sampLoc.append_trace(go.Scatter(x=ld.df_samploc_av.index,y=ld.df_samploc_av['Latitude..DD.'].to_numpy(), name='',
                                    mode='lines+markers', marker_symbol="circle", marker=dict(color=req.colors['marker'], 
                                                                                              size=6, line=dict(width=0)),
                                    line=dict(color=req.colors['linecolor'], width=2)), row=1, col=1)
fig_sampLoc.append_trace(go.Scatter(x=ld.df_samploc_av.index,y=ld.df_samploc_av['Longitude..DD.'].to_numpy(), name='', 
                                    mode='lines+markers', marker_symbol="circle", marker=dict(color=req.colors['marker'], 
                                                                                              size=6, line=dict(width=0)),
                                    line=dict(color=req.colors['linecolor'], width=2)), row=2, col=1)
fig_sampLoc.append_trace(go.Scatter(x=ld.df_samploc_av.index,y=ld.df_samploc_av['salinity'].to_numpy(), name='',
                                    mode='lines+markers', marker_symbol="circle", marker=dict(color=req.colors['marker'], 
                                                                                              size=6, line=dict(width=0)),
                                    line=dict(color=req.colors['linecolor'], width=2)), row=3, col=1)
fig_sampLoc.append_trace(go.Scatter(x=ld.df_min.index,y=ld.df_min['salinity'].to_numpy(), mode='lines', name='',
                                    line_color=req.colors['linecolor'], line=dict(color=req.colors['linecolor'], width=0), 
                                    connectgaps=False, fill='tonexty', fillcolor=req.colors['fill_color'],), row=3, col=1)
fig_sampLoc.append_trace(go.Scatter(x=ld.df_min.index,y=ld.df_max['salinity'].to_numpy(), mode='lines', name='',
                                    line_color=req.colors['linecolor'], line=dict(color=req.colors['linecolor'], width=0),  
                                    connectgaps=False, fill='tonexty', fillcolor=req.colors['fill_color'],), row=3, col=1)

fig_sampLoc.data[0].line.dash = 'dash'
fig_sampLoc.data[1].line.dash = 'dash'
fig_sampLoc.data[2].line.dash = 'dash'
fig_sampLoc['layout']['yaxis']['autorange'] = "reversed"

fig_sampLoc.update_xaxes(linecolor=req.colors['linecolor'], row=1, col=1)
fig_sampLoc.update_xaxes(linecolor=req.colors['linecolor'], row=2, col=1)
fig_sampLoc.update_xaxes(title_text="Station Name", linecolor=req.colors['linecolor'], row=3, col=1)
fig_sampLoc.update_yaxes(title_text="Latitude", linecolor=req.colors['linecolor'], row=1, col=1)
fig_sampLoc.update_yaxes(title_text="Longitude", linecolor=req.colors['linecolor'], row=2, col=1)
fig_sampLoc.update_yaxes(title_text="Salinity", linecolor=req.colors['linecolor'], row=3, col=1)
fig_sampLoc.update_layout(plot_bgcolor=req.colors['background'], font=dict(color=req.colors['font'], size=req.fs_label), 
                          hovermode="x", showlegend=False, width=450, height=470, margin={"l": 30, "r": 30, "t": 20, "b": 100})


# ..................................................................
# FIGURE 2 · sampling location MAP
# Add some markers or other features to the Folium map if needed
latitude, longitude = ld.ls_sampling_loc_all[:, 0], ld.ls_sampling_loc_all[:, 1]
folium_map, dic_marker = dbs.foliumMap(file=req.url, center=req.map_center, zoom_start=7.25, ls_label_plankton=dbs.ls_phyto, 
                                       ls_label_parameter=dbs.ls_label_parameter, df_sampling=ld.df_samp, miniMap=True,
                                       tiles='cartodbpositron')
# Convert Folium map to HTML
folium_map.save('foliumMap_LakeAid_Greenland.html')
# Create a Plotly Scatter Mapbox trace
latitude, longitude = ld.ls_sampling_loc_all[:, 0], ld.ls_sampling_loc_all[:, 1]
scatter_mapbox_trace = go.Scattermapbox(lat=latitude, lon=longitude, mode='markers', marker=dict(size=14, color='blue'),
                                        text=[dic_marker[k][1] for k in dic_marker.keys()])
# Create a Plotly layout with Mapbox
layout = go.Layout(autosize=True, hovermode='closest', 
                   mapbox=dict(style="mapbox://styles/mapbox/streets-v11",
                accesstoken='sk.eyJ1IjoiZW52aXBhdGFibGUiLCJhIjoiY2xzMzdmbmw0MHB5ODJycXBrZGxkMHhwNSJ9.noTVlQqXNVjFHD5cJYf7NA',  
                bearing=10, center=dict({'lat':req.map_center[0], 'lon':req.map_center[1]}), pitch=0, zoom=7.3),)

# Create a Plotly Figure
figGeoMap = go.Figure(data=[scatter_mapbox_trace], layout=layout)
figGeoMap.update_layout(updatemenus=[dict(type='buttons', x=1.05,y=0.7, buttons=[dict(label='Show Folium Map', method='relayout', 
                                                                                      args=['geo.visible', False])],)])


# ..................................................................
# FIGURE 4 · Correlation HeatMap - Image
dic_fig_corr = dict()
for fi in ld.dic_file_corr.keys():
    fig_corr = go.Figure()
    # constants
    img_width, img_height = 500, 400
    # add invisible scatter trace – to help the autoresize logic work.
    fig_corr.add_trace(go.Scatter(x=[0, img_width], y=[0, img_height], mode="markers", marker_opacity=0))
    fig_corr.update_xaxes(visible=False, range=[0, img_width])
    fig_corr.update_yaxes(visible=False, range=[0, img_height], scaleanchor="x")

    # add image and custommize layout
    fig_corr.add_layout_image(dict(x=0, sizex=img_width, y=img_height, sizey=img_height, xref="x", yref="y", opacity=1.0,
                                   layer="below", sizing="stretch", source=Image.open(ld.dic_file_corr[fi]) ))
    fig_corr.update_layout(width=img_width, height=img_height, margin={"l": 0, "r": 0, "t": 0, "b": 0},)
    dic_fig_corr[fi] = fig_corr


# .................................................................................................................................
# DYNAMIC FIGURES · CALLBACK FUNCTION
# .................................................................................................................................
def depth_profile(para):
    # get the equivalent parameter (forend->backend)
    para_trans = ld.dic_trans[para]
    xlabel = ld.create_label(para)

    # define color-palette and zoom range 
    ls_color, zoom = dbs.helpers_depthProf(df_plankton=ld.df_plank, para=para_trans, 
                                           ls_stationP=ld.dic_prep['station_wPlank_sorted'], color_map=req.colors['divMap'])

    # initiate figure window
    fig = go.Figure()
    en = 0
    for p in ld.dic_prep['station_wPlank_sorted']:
        df1 = ld.df_plank[ld.df_plank['Station_name'] == p][['Sample_depth']+dbs.dic_para[1]+dbs.dic_para[2]]
        df1 = df1.groupby(para_trans).mean().sort_values('Sample_depth')
        
        if df1.shape[0] > 2:
            # interpolation for overall fitting at sampling station
            x, y = df1.index.to_numpy(), df1['Sample_depth'].to_numpy()
            tck, u = splprep([x, y], s=5)
            new_points = splev(u, tck)

            color = matplotlib.colors.to_hex(ls_color(en), keep_alpha=False)
            fig.add_trace(go.Scatter(x=x,y=y, mode='markers', showlegend=False, marker_symbol="circle", 
                                     marker=dict(color=color, size=6, line=dict(width=0)),))
            fig.add_trace(go.Scatter(x=new_points[0], y=new_points[1], mode='lines', name=p, line_color=color, 
                                     line=dict(color=color, width=2),)) 
            en+=1
        elif df1.shape[0] <= 2:
            # only plot the sampling station info
            x, y = df1.index.to_numpy(), df1['Sample_depth'].to_numpy()
            color = matplotlib.colors.to_hex(ls_color(en), keep_alpha=False)
            fig.add_trace(go.Scatter(x=x,y=y, mode='markers', name=p, marker_symbol="circle", 
                                     marker=dict(color=color, size=6, line=dict(width=0)),))
            en+=1
        # horizontal line to mark sea level
        fig.add_hline(y=0., line_dash="dash", line=dict(color=req.colors['spine'], width=1))

    # update layout
    if zoom and 'x' in zoom:
        fig.update_xaxes(title_text=xlabel, linecolor=req.colors['linecolor'], range=zoom['x'])
    else:
        fig.update_xaxes(title_text=xlabel, linecolor=req.colors['linecolor'])
    if zoom and 'y' in zoom:
        fig.update_yaxes(title_text='Sample Depth, m', linecolor=req.colors['linecolor'], range=zoom['y'])
    else:
        fig.update_yaxes(title_text='Sample Depth, m', linecolor=req.colors['linecolor'])

    fig['layout']['yaxis']['autorange'] = "reversed"
    fig.update_layout(xaxis={'side': 'top'}, height=650, width=450, plot_bgcolor=req.colors['background'], 
                      font=dict(color=req.colors['font'], size=req.fs_label2), legend_title_text='Station name',
                      legend=dict(x=0.95, y=0.05, traceorder="reversed", font=dict(size=req.fs_label, color=req.colors['font']), 
                                  bgcolor=req.colors['background']), margin={"l": 30, "r": 50, "t": 100, "b": 25})

    return fig


def depth_profileDetail(para):
    # get the equivalent parameter (forend->backend)
    para_trans = ld.dic_trans[para]
    xlabel = ld.create_label(para)

    # calculate median and deviation of Depth Profile of selected parameter
    label = 'Sample_depth'
    df_ = ld.df_plank[[label, para_trans]]
    df_DPmedian = df_.set_index(label).sort_index().dropna().groupby(label).median()
    df_DPstd = df_.set_index(label).sort_index().dropna().groupby(label).std()
    df_DPmin = df_DPmedian-df_DPstd
    df_DPmax = df_DPmedian+df_DPstd

    # define color-palette and zoom range 
    ls_color, zoom = dbs.helpers_depthProf(df_plankton=ld.df_plank, para=para_trans, 
                                           ls_stationP=ld.dic_prep['station_wPlank_sorted'], color_map=req.colors['divMap'])

    # initiate figure window
    fig = go.Figure()
    # horizontal line to mark sea level
    fig.add_hline(y=0., line_dash="dash", line=dict(color=req.colors['spine'], width=1))
    # average depth profile and fill between STD
    fig.add_trace(go.Scatter(x=df_DPmedian[para_trans].to_numpy(),y=df_DPmedian.index, mode='lines', showlegend=False,
                             line_dash="dash", line=dict(color='black', width=2),fill=None))
    fig.add_trace(go.Scatter(x=df_DPmin[para_trans].dropna().values, y=df_DPmin[para_trans].dropna().index, mode='none',
                             showlegend=False, line_color=req.colors['linecolor'], fillcolor=req.colors['fill_color'],
                             line=dict(color=req.colors['linecolor'], width=0), connectgaps=False, fill='tonexty',))
    fig.add_trace(go.Scatter(x=df_DPmax[para_trans].dropna().values, y=df_DPmax[para_trans].dropna().index, mode='none',
                             showlegend=False, line_color=req.colors['linecolor'], fillcolor=req.colors['fill_color'],
                             line=dict(color=req.colors['linecolor'], width=0), connectgaps=False, fill='tonexty',))

    fig['layout']['yaxis']['autorange']="reversed"
    fig.update_xaxes(title_text=xlabel, linecolor=req.colors['linecolor'])
    fig.update_yaxes(title_text='Sample Depth, m', linecolor=req.colors['linecolor'])
    fig.update_layout(uniformtext_minsize=9, uniformtext_mode='hide', height=650, width=350, plot_bgcolor=req.colors['background'],
                      font=dict(color=req.colors['font'], size=req.fs_label2), xaxis={'side': 'top'}, 
                      margin={"l": 30, "r": 75, "t": 100, "b": 25})
    return fig


def depthCorrelation(lbl_plankton):
    # correct labeling due to inconistency
    if lbl_plankton =='Bacteria.HNF':
        lbl_plankton = 'Bac.HNF'
    # detailed info about sampling depth with idenfitied correlation >0.5 or <-0.5
    df_test = ld.dic_Scorr[lbl_plankton].set_index('Sample depth (#_samples)')

    # ------------------------------------------------------------------------------
    fig = go.Figure()
    fig.update_xaxes(title_text="Cummulative Correlation Factor")
    fig.update_yaxes(title_text="Sample Depth, m")

    ls_legend = list()
    for en, ind in enumerate(df_test.index):
        # get a list of parameters that correlate with the plankton at that level
        ls_values = df_test.loc[df_test.index[en]].dropna().to_numpy()
        if len(ls_values) >= 1:
            for v in ls_values:
                if v.split(' ')[0] not in ['conductivity', 'NO2', 'NO3']:
                    color, val, name, grp = dbs.helper_plot(v)
                    if name in ls_legend:
                        showlegend=False
                    else:
                        showlegend=True
                        ls_legend.append(name)
                    # plot the bars and add parameter information as annotation
                    label = v.split(' ')[0]
                    if label in ld.dic_reTrans.keys():
                        label = ld.dic_reTrans[label]
                    fig.add_trace(go.Bar(y=[en], x=[np.abs(val)], width=0.5, orientation='h', name=name, text=label, 
                                        textposition='auto',  showlegend=showlegend, 
                                        marker=dict(color=color, line=dict(color=req.colors['background'], width=0.5)),))
    fig.update_layout(plot_bgcolor=req.colors['background'], font=dict(color=req.colors['font'], size=req.fs_label), 
                      height=650, margin={"l": 2, "r": 10, "t": 80, "b": 2}, barmode='relative', uniformtext_minsize=9, 
                      uniformtext_mode='hide', legend=dict(y=1.05, x=0.25, orientation="h"), xaxis=dict(showline=False), 
                      yaxis=dict(showgrid=False, zeroline=False, tickmode='array', showline=False, tickvals=np.arange(9), 
                                 ticktext=df_test.index, autorange="reversed"),
                      ) # width=350, 
    return fig


def depthCorrelationDetail(lbl_plankton):
    unit = ld.dic_units[lbl_plankton][1]
    # correct labeling due to inconistency
    if lbl_plankton =='Bacteria.HNF':
        lbl_plankton = 'Bac.HNF'
    
    # detailed info about sampling depth with idenfitied correlation >0.5 or <-0.5 
    df_test = ld.dic_Scorr[lbl_plankton].set_index('Sample depth (#_samples)')
    ls_depth = [int(i.split(' ')[0]) for i in df_test.index]
    df_plk = ld.dic_Scorr[lbl_plankton]
    df_mask = df_plk.copy()
    for cig in ld.ls_ignoreCorr:
        df_mask = ld.maskParameter(df=df_mask, ls_col=df_plk.columns[1:], cig=cig) 
    df_mask = df_mask.dropna(axis=1, how='all')
 
    ncols, nrows = df_plk.shape[1]-1, df_plk['Unnamed: 1'].dropna().shape[0]
    ncols1, nrows1 = df_mask.shape[1]-1, df_mask['Unnamed: 1'].dropna().shape[0]
    if df_mask[df_mask.columns[1:]].dropna(axis=0, how='all').shape[0] != df_plk[df_plk.columns[1:]].dropna(axis=0, 
                                                                                                            how='all').shape[0]:
        N = nrows1
    else:
        N = nrows
            
    # dynamic adjustment of image height depending on number of subplots/nrows
    if N <5:
        height = 600
    elif N <=5:
        height = 650
    elif N ==6:
        height = 750
    elif N == 7:
        height = 850
    else:
        height = 1000

    # Specify the phytoplankton unit
    title = 'In Detail · {} (in {}) vs. Environmental Parameters Correlation'.format(lbl_plankton, unit)

    # ------------------------------------------------------------------------------
    fig = make_subplots(rows=N, cols=ncols1, horizontal_spacing=0.095, vertical_spacing=0.05, 
                        subplot_titles=['plot·'+str(i) for i in range(ncols1*N)])
    d, ax = 0, 0
    for i in range(df_mask.shape[0]):
        if len(df_mask.loc[i].dropna().to_numpy()[1:]) > 0:
            for w in range(ncols1):
                if w in range(len(df_mask.loc[i].dropna().to_numpy()[1:])):
                    # get the data and axis labels
                    val, p, label = dbs.helper_detailCorr(df_plk=df_mask, i=i, w=w)
                    if p in ld.dic_reTrans.keys():
                        pi = ld.dic_reTrans[p]
                    else:
                        pi = p
                    suptitle = pi + ' | ' + label.split('· ')[0][0] + '~' + label.split(' · ')[1].strip()
                    fig.layout.annotations[ax].update(text=suptitle)

                    # plot the data
                    xdata, ydata = ld.dic_df[ls_depth[i]][lbl_plankton].to_numpy(), ld.dic_df[ls_depth[i]][p].to_numpy()
                    fig.append_trace(go.Scatter(x=xdata, y=ydata, name=p + ' ' + label, mode='markers', marker_symbol="circle", 
                                                marker=dict(color=req.colors['marker'], size=6, line=dict(width=0)),), 
                                    row=d+1, col=w+1)
                    
                    # plot the (linear) correlation
                    reg = LinearRegression().fit(xdata.reshape(-1, 1), ydata)
                    xnew = np.linspace(np.min(xdata), np.max(xdata))
                    ynew = xnew*reg.coef_[0] + reg.intercept_
                    fig.append_trace(go.Scatter(x=xnew, y=ynew, mode='lines', line=dict(color=req.colors['marker'], dash='dot',
                                                                                        width=1)), row=d+1, col=w+1)
                    ax +=1
                else:
                    fig['layout']['annotations'][ax].update(text='')
                    ax +=1
            d +=1
    
    if df_mask[df_mask.columns[1:]].dropna(axis=0, how='all').shape[0] != df_plk[df_plk.columns[1:]].dropna(axis=0,
                                                                                                            how='all').shape[0]:
        # last row - remove title
        for f in fig['layout']['annotations'][nrows1*ncols1:]:
            f.update(text='')
    
    fig["layout"]["annotations"][0]["font"]["size"] = req.fs_label
    fig["layout"]["annotations"][1]["font"]["size"] = req.fs_label
    fig["layout"]["annotations"][2]["font"]["size"] = req.fs_label
    fig.update_xaxes(linecolor=req.colors['linecolor'], tickangle=0) 
    fig.update_yaxes(linecolor=req.colors['linecolor'])
    fig.update_annotations(dict(font_size=req.fs_label))
    fig.update_layout(uniformtext_minsize=req.fs_text, uniformtext_mode='hide', title_text=title, title_font=dict(size=req.fs_label),
                      plot_bgcolor=req.colors['background'], font=dict(color=req.colors['font'], size=req.fs_label), height=height, 
                      margin={"l": 10, "r": 10, "t": 80, "b": 2}, yaxis=dict(tickfont=dict(size=9)), showlegend=False,) #width=650, 
    fig.update_annotations(yshift=-5)
    return fig


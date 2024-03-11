import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import seaborn as sns
import folium
from folium import plugins

import colormaps as cmaps
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy.interpolate import splprep, splev
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from datetime import datetime
import os
import math


# Gobal options
plt.rcParams['font.family'] = 'serif'
sns.set_style('ticks')

# speficy directories for loading and exporting data
file = '/Users/au652733/Library/CloudStorage/OneDrive-AarhusUniversitet/00 Privat/03 Work/03 Freelance/04 DS-projects/202307_LakeAid/data/Greenland-fjord_FieldData.xlsx'


# ............................................................................................................
# parameters for geomapping
ls_geoMap = ['Station_name', 'DD.MM.YYY', 'Latitude..DD.', 'Longitude..DD.', 'Sample_depth']

# parameter selection / clustering
ls_phyto = ['Picoplankton', 'Nanoplankton', 'HNA', 'LNA', 'Bacteria', 'HNA.LNA', 'HNF', 'Bac.HNF']
# relevant/curious and enough data
ls_paraGen = ['temperature', 'conductivity', 'fluorescence', 'par', 'oxygen', 'turbidity', 'pH', 'salinity']
ls_para1 = ['NO2NO3', 'NO2', 'NO3', 'Chla_GFF', 'NH3', 'PO4', 'Si']

# only few data available. might be to little to say anything about correlation
ls_para2 = ['DOC_uM', 'TN_uM', 'PON (µM)', 'POC (µM)', 'C.N', 'd15N', 'd13C']
ls_para3 = ['TSS/inorg particles (mg/L)', 'Org. Paticulate matter (Ig method) mg/L', 'PIC:POC ratio']

# store parameters in dictionary
dic_para = dict({0: ls_phyto, 1: ls_paraGen, 2: ls_para1, 3: ls_para2, 4: ls_para3})

# ....................................
## STATION NAMES VS LONGITUDE/LATITUDE
ls_loc = ['Station_name', 'Latitude..DD.', 'Longitude..DD.', 'salinity']
# manually sorted station names >> from inside to outside (into the fjord)
ls_loc_sort = ['Glacial Ice', 'Tyroler River', 'Tyrolerfjord Plume', 'TYRO_01', 'TYRO_05', 'Zackenberg River', 
               'YS_3.14', 'Standard station ','GH_05']

ls_label_parameter = ['Sample_depth', 'temperature', 'conductivity', 'fluorescence', 'par', 'oxygen',
                      'turbidity', 'pH', 'scan', 'depth', 'salinity', 'flag']

# ............................................................................................................
# import and preprocess data
def load_data(url):
    url='https://drive.google.com/uc?id=' + url.split('/')[-2]
    df = pd.read_excel(url)
    return df 


def preprocess_data(df):
    # retrieve station names
    ls_station = list(dict.fromkeys(df['Station_name'].dropna()))
    ls_station1 = list(dict.fromkeys([p if isinstance(p, int) else p.split('_')[0].strip() for p in ls_station]))

    # sampling data set only where plankton was found
    df_plankton = df.loc[df[ls_phyto].dropna().index]
    
    # relative plankton abundance
    df_relAbund = (100*df_plankton[ls_phyto].T / df_plankton[ls_phyto].sum(axis=1)).T
    # parameters without plankton
    df_para = df_plankton[[c for c in df_plankton.columns if c not in ls_phyto]]
    # df_plankton1 - relative abundance
    df_plankton1 = pd.concat([df_para, df_relAbund], axis=1)
    # sample depths where plankton was found (across different stations)
    ls_sDepth = sorted(df_plankton1['Sample_depth'].drop_duplicates().to_list())

    # station names with plankton sorted from inland to open water
    ls_stationP = df_plankton.sort_values('Longitude..DD.')['Station_name'].drop_duplicates().to_numpy()[:-1] 
    
    # summarize in one dictionary
    dicData_prep = dict({'station_wPlankton': df_plankton, 'rel_abundance': df_relAbund, 'station_envPara': df_para, 
                         'dataAll_wRelAbun': df_plankton1, 'station_wPlank_sorted': ls_stationP, 'depth_wPlankton': ls_sDepth})
    return dicData_prep


def prep_map(file):
    df1 = load_data(file)
    dic_prep = preprocess_data(df1)

    # DataPrep · geographical mapping
    df_samp = dic_prep['station_wPlankton'][ls_geoMap].dropna()
    # get the sampling location
    ls_sampling_loc = df_samp[['Latitude..DD.','Longitude..DD.']].drop_duplicates().to_numpy()
    # remaining stations where no phytoplankton was found
    ls_sampling_loc_all = df1[['Latitude..DD.','Longitude..DD.']].dropna().drop_duplicates().to_numpy()
    ls_sampling_loc_remain = [list(item) for item in ls_sampling_loc_all if item not in ls_sampling_loc]
    
    return df1, ls_sampling_loc, ls_sampling_loc_remain


# ............................................................................................................
def samplingLocation(df_plank1):
    # get the average/std of salinity at the sampling station
    df_grp_av = df_plank1[ls_loc].groupby('Station_name').mean()
    df_grp_std = df_plank1[ls_loc].groupby('Station_name').std()

    # sort from inside to outside (into the fjord)
    df_grp_av = df_grp_av.loc[ls_loc_sort]
    df_grp_std = df_grp_std.loc[ls_loc_sort]

    # min/max for ranging the salinity
    df_min = df_grp_av-df_grp_std.replace(np.nan, 0)
    df_max = df_grp_av+df_grp_std.replace(np.nan, 0)
    df_min = df_min.loc[df_grp_av.index]
    df_max = df_max.loc[df_grp_av.index]

    return df_grp_av, df_grp_std, df_min, df_max


def corr_matrix(df):
    return dict(map(lambda m: (m, df[['Sample_depth', 'Longitude..DD.']+ls_paraGen+ls_para1+ls_phyto].corr(method=m)), 
                    ['spearman', 'pearson']))


def get_stationInfo(df_sampling, p):
    df_samplingP = df_sampling[df_sampling['Latitude..DD.'] == p[0]]
    df_samplingP = df_samplingP[df_samplingP['Longitude..DD.'] == p[1]]

    name = df_samplingP['Station_name'].drop_duplicates().to_numpy()[0]
    date = pd.to_datetime(str(df_samplingP['DD.MM.YYY'].drop_duplicates().to_numpy()[0])).strftime('%d/%m/%Y')
    return name, date


def find_availableParameters(df, ls_label_parameter, p):
    dfP = df[df['Latitude..DD.'] == p[0]]
    ls_para_SampSite = dfP[dfP['Longitude..DD.'] == p[1]].T.dropna().index.to_list()

    ls_para_available = list()
    for pa in ls_label_parameter:
        if pa in ls_para_SampSite:
            ls_para_available.append(pa)
    return ls_para_available


def found_plankton(df, name, date, ls_label_plankton):
    dfP = df[df['Station_name'] == name]
    d_split = date.split('/')
    ls = dfP[dfP['DD.MM.YYY'] == d_split[-1]+'-'+d_split[1]+'-'+d_split[0]][ls_label_plankton].dropna().columns
    return ls


# ............................................................................................................
def helper_plot(v):
    v_nmb = v.split('(')[1].split(')')[0]
    label = v.split(' ')[0]
    # make some labels shorter
    if 'Longitude' in label:
        label = 'Longitude'

    # separate whether it is pearson or spearman -> color coding
    if 'p;' in v:
        color = '#00435A'
        val = float(v_nmb.split('; ')[1])
        name = 'pearson'
        grp = 1
    else:
        color = 'lightgrey'
        val = float(v_nmb)
        name = 'spearman'
        grp = 2
    return color, val, name, grp


def helper_detailCorr(df_plk, i, w):
    val = df_plk.loc[i].dropna().to_numpy()[1:][w]
    if 'p' in val.split(' (')[1]:
        label = 'Pearson · ' + val.split(' (p;')[1].split(')')[0]
    else:
        label = 'Spearman · ' + val.split(' (')[1].split(')')[0]
    p = val.split(' (')[0]
    if 'PAR' == p:
        p = 'par'
    return val, p, label


def helpers_depthProf(df_plankton, para, ls_stationP, color_map):
    # how many colors are needed
    # loop through list of stations and count appearance, where more than 1 sample point has been measured
    count = 0
    for p in ls_stationP:
        if df_plankton[df_plankton['Station_name'] == p].shape[0] >= 1:
            count +=1
    ls_colors = color_map.discrete(count) # sns.color_palette(color_map, count)

    # specify zoom level depending on parameter (adjusting the scale; subjective decision)
    if para == 'salinity' or para == 'temperature':
        zoom = dict({'y': (-.5, 15)})
    elif para == 'turbidity':
        zoom = dict({'x': (-2., 20)})
    else:
        zoom = None
    return ls_colors, zoom


def foliumMap(file, center, zoom_start, ls_label_parameter, ls_label_plankton, df_sampling, tiles='cartodbpositron', 
              miniMap=True):
    df, ls_sampling_loc, ls_sampling_loc_remain = prep_map(file)

    # initialize the map
    m = folium.Map(location=center,tiles=tiles, zoom_start=zoom_start)
    #, zoom_control=False, scrollWheelZoom=True, dragging=False)

    # add a minimap for user adjustments (ease of use)
    if miniMap is True:
        minimap = plugins.MiniMap()
        m.add_child(minimap)

    # -----------------------------------------------------------------------
    # create marker points where phytoplankton WAS found
    for p in ls_sampling_loc:
        # station infos
        name, date = get_stationInfo(df_sampling=df_sampling, p=p)
        ls_para_available = find_availableParameters(df=df, ls_label_parameter=ls_label_parameter, p=p)

        # get which plankton was found at a specific sampling station
        ls_plankton_found = list(found_plankton(df=df, name=name, date=date, ls_label_plankton=ls_label_plankton))


        # create marker circle
        popup_text = "<h4><b>Measurement Overview</b></h4>"+ "<b>Station Name · </b>"+str(name)+ "<br><b>Date · </b>"+\
            str(date) + "<br><b><p style=color:#BF4D34>-- Plankton found --</b></p><b>Additional_parameters</b><br> · " +\
                str('<br>· '.join(ls_para_available))
        iframe = folium.IFrame(popup_text)        
        popup = folium.Popup(iframe, min_width=300, max_width=350)
        marker = folium.CircleMarker((p[0], p[1]), radius=5, popup=popup, color="#BF4D34", fill=True, fill_color="#BF4D34",
                                     fill_opacity=.85).add_to(m)#
    # create marker points for REMAINING stations
    dic_marker = dict()
    for p in ls_sampling_loc_remain:
        # station infos
        name, date = get_stationInfo(df_sampling=df, p=p)    
        ls_para_available = find_availableParameters(df=df, ls_label_parameter=ls_label_parameter, p=p)

        # create marker circle
        popup_text = "<h4><b>Measurement Overview</b></h4>"+ "<b>Station Name · </b>"+str(name)+ "<br><b>Date · </b>"+\
            str(date) + "<br><b><p style=color:#002333>!!! No plankton found</b></p><b>Additional_parameters</b><br> · " +\
                str('<br>· '.join(ls_para_available))
        iframe = folium.IFrame(popup_text)        
        
        popup = folium.Popup(iframe, min_width=300, max_width=350)
        marker = folium.CircleMarker((p[0], p[1]), radius=5, popup=popup, color="none", fill=True, fill_color="#002333",
                                     fill_opacity=0.6).add_to(m)
        dic_marker[tuple(p)] = (marker, popup_text)
    return m, dic_marker


def plot_correlDepthProfile(lbl_plankton, df_test, fgs=(5,5), fsT=8, fs=10):
    fig, ax = plt.subplots(figsize=fgs)
    ax.set_title(lbl_plankton, fontsize=fs*1.15)
    ax.set_ylabel('Sample Depth, m', fontsize=fs*1.05)
    ax.set_xlabel('Cummulative correlation factor', fontsize=fs*1.05)
    
    add = False
    ls_legend, ls_handles = list(), list()
    for ind in range(len(df_test.index)):
        # get a list of parameters that correlate with the plankton at that level
        ls_values = df_test.loc[df_test.index[ind]].dropna().to_numpy()

        if len(ls_values) >= 1:
            w = 0
            for v in ls_values:
                # identify correlating parameter and the correlation factor
                v_nmb = v.split('(')[1].split(')')[0]
                label = v.split(' ')[0]
                # make some labels shorter
                if 'Longitude' in label:
                    label = 'Longitude'

                # separate whether it is pearson or spearman -> color coding
                if 'p;' in v:
                    color, colorT = '#00435A', 'white'
                    v = float(v_nmb.split('; ')[1])
                    add = True if 'pearson' not in ls_legend else False
                    if 'pearson' not in ls_legend:
                        ls_legend.append('pearson')
                else:
                    color, colorT = 'lightgrey', 'k'
                    v = float(v_nmb)
                    add = True if 'spearman' not in ls_legend else False
                    if 'spearman' not in ls_legend:
                        ls_legend.append('spearman')

                # plot the bars and add parameter information as annotation
                ln1, = ax.barh([df_test.index[ind]], [np.abs(v)], height=0.5, color=color, left=w, lw=0.5)
                plt.text(w+.05, ind, label, fontweight="bold", fontsize=fsT, color=colorT, ha='left', va='center')

                # legend handling -> add axis when a new legend info was added (new pearson/spearman)
                if add is True:
                    ls_handles.append(ln1)
                w +=np.abs(v)
        else:
            ax.barh([df_test.index[ind]], [0], height=0.5, color='lightgrey')

    # legend handles, labels
    ax.legend(ls_handles, ls_legend, fontsize=fs*0.75, loc='lower right')
    # axis layout
    ax.invert_yaxis()
    ax.tick_params(labelsize=fs)
    sns.despine()
    plt.tight_layout()
    return fig

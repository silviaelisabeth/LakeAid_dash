import requirements as req
import dbs_Dashboard as dbs

import pandas as pd
from glob import glob


# ..............................................................................................
# text and headings
title = 'Greenland Excursion | Environmental Parameters vs. Plankton Groups'
heading1 = 'Depth Profiles · Environmental Parameters'
heading2 = 'Correlation Information · Environmental Parameters'
heading3 = 'Correlation Profiles Phytoplankton'
impressum = '2024© Plotly made by Silvia E. Zieger. All rights reserved. Email · info@silviazieger.com | Website · www.silviazieger.com '
      
# parameter to be selected via RadioItems              
ls_enviPara = ['Turbidity', 'Conductivity', 'Temperature', 'Light', 'Salinity', 'Oxygen', 'pH', 'Total-Chlorophyll', 'Fluorescence',
               'Total-Nitrogen', 'Nitrite', 'Nitrate', 'Ammonium', 'Phosphate', 'Silicate',]
ls_corrMeth = ['Pearson', 'Spearman']
ls_phyto = ['Picoplankton', 'Nanoplankton', 'HNA', 'LNA', 'HNA.LNA', 'Bacteria', 'HNF', 'Bacteria.HNF']

# translate parameter from RadioItems to parameter available in data set
dic_trans = {'Temperature':'temperature', 'Conductivity':'conductivity', 'Fluorescence':'fluorescence', 'Light':'par',
             'Oxygen':'oxygen', 'Turbidity':'turbidity', 'pH':'pH', 'Salinity':'salinity', 'Total-Nitrogen':'NO2NO3',
             'Nitrite':'NO2', 'Nitrate':'NO3', 'Total-Chlorophyll':'Chla_GFF', 'Ammonium': 'NH3', 'Phosphate':'PO4', 'Silicate':'Si'}

dic_reTrans = {'NO2NO3': 'Total-N', 'Longitude..DD.': 'Longitude', 'turbidity': 'Turbidity', 'temperature':'Temperature',
               'conductivity':'Conductivity', 'fluorescence':'Fluorescence', 'par':'Light', 'oxygen':'Oxygen', 'salinity':'Salinity', 
               'NO2':'Nitrite', 'NO3':'Nitrate', 'Chla_GFF':'Total-Chl', 'NH3':'Ammonium', 'PO4':'Phosphate', 'Si':'Silicate'
               }

# retrieve respective units and formula
dic_units = {'Oxygen':('O<sub>2</sub>','μmol&#8201;kg<sup>-1</sup>'), 'Turbidity':(None,'FTU'), 'pH':(None,None), 
             'Fluorescence':(None,'rfu'), 'Temperature':(None,'degC'), 'Conductivity':(None,'Sm<sup>-1</sup>'), 
             'Light':(None,'μmol&#8201;m<sup>-2</sup>&#8201;s<sup>-1</sup>'), 'Salinity':(None,'PSS'), 
             'Total-Nitrogen':(None,'μM'), 'Nitrite':('NO<sub>2</sub>','μM'), 'Nitrate':('NO<sub>3</sub>','μM'), 
             'Total-Chlorophyll':(None,'ppm'), 'Silicate':('Si','μM'), 'Ammonium':('NH<sub>3</sub>','μM'), 
             'Phosphate':('PO<sub>4</sub>','μM'), 
             'Picoplankton':(None,'cells&#8201;mL<sup>-1</sup>'), 'Nanoplankton':(None,'cells&#8201;mL<sup>-1</sup>'), 
             'HNA':(None,'cells&#8201;mL<sup>-1</sup>'), 'LNA':(None,'cells&#8201;mL<sup>-1</sup>'), 'HNA.LNA':(None,'ratio'),
             'Bacteria':(None,'cells&#8201;mL<sup>-1</sup>'), 'HNF':(None,'cells&#8201;mL<sup>-1</sup>'), 
             'Bacteria.HNF':(None,'cells&#8201;mL<sup>-1</sup>')}

# masking a list of parameters | conducitivity, NO2, and NO3
ls_ignoreCorr = ['conductivity', 'NO2(<!NO3)', '(<!NO2)NO3']


# Disable the autosize on double click because it adds unwanted margins around the image
# Mor e detail: https://plotly.com/python/configuration-options/
dic_descrip = dict({'Pearson': 'Pearson correlation is the most common way of measuring a linear correlation. It is a number between –1 and 1 that measures the strength and direction of the relationship between two variables.', 'Spearman': 'Spearman is a nonparametric alternative to Pearson’s correlation for data that follow curvilinear, monotonic relationships and for ordinal data. It is a number between –1 and 1 that measures the strength and direction of the relationship between two variables.'})

# ..............................................................................................
# data and preprocessing
df1 = dbs.load_data(req.url)
dic_prep = dbs.preprocess_data(df1)
df_plank = dic_prep['dataAll_wRelAbun']

# correlation matrix (spearman and pearson) for relative abundance
dic_corr = dbs.corr_matrix(df=df_plank)
# correlation per sampling depth
dic_Scorr = pd.read_excel(req.file, sheet_name=None)
# specifics for sampling location where plankton was found 
df_samploc_av, df_samploc_std, df_min, df_max = dbs.samplingLocation(df_plank)

# DataPrep · geographical mapping
df_samp = dic_prep['station_wPlankton'][dbs.ls_geoMap].dropna()
# get the sampling location
ls_sampling_loc = df_samp[['Latitude..DD.','Longitude..DD.']].drop_duplicates().to_numpy()
# remaining stations where no phytoplankton was found
ls_sampling_loc_all = df1[['Latitude..DD.','Longitude..DD.']].dropna().drop_duplicates().to_numpy()
ls_sampling_loc_remain = [list(item) for item in ls_sampling_loc_all if item not in ls_sampling_loc]

# helper function for depth profile of different environmental parameters
dic_df = dict(map(lambda d: (d, df_plank[df_plank['Sample_depth'] == d]), dic_prep['depth_wPlankton']))

# load images for correlation plots/heatmaps
ls_file_corr = [f for f in glob(req.path_save + req.file_corr_pattern)]
dic_file_corr = dict(map(lambda fcorr: (fcorr.split('/')[-1].split(req.file_corr_pattern.split('.')[0][1:-1])[0].split('_')[1],
                                        fcorr), ls_file_corr))


def create_label(para):
    unit = dic_units[para][1]
    if dic_units[para][0]:
        formula = dic_units[para][0]
        label = para + ' ('+ formula + '), ' + unit
    else:
        if unit:
            label = para + ', ' + unit
        else:
            label = para
    return label


def maskParameter(df, ls_col, cig):
    df_mask = pd.concat([df[c][df[c].str.contains(cig, na=False) == False] for c in ls_col], axis=1)
    df_mask = pd.concat([df['Sample depth (#_samples)'], df_mask], axis=1)
    return df_mask
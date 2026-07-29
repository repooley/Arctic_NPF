# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 10:29:39 2026

@author: repooley
"""

import icartt
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from scipy.stats import binned_statistic_2d
from datetime import date
import matplotlib.ticker as ticker

###################
##--User inputs--##
###################

##--Set the base directory to project folder--##
directory = r"C:\Users\repooley\REP_PhD\Arctic_NPF\ATom2018\data"

##--The flight10 file also includes flight11--##
flights_to_analyze = ["Flight2", "Flight10", "Flight11", "Flight12"]

##--Set binning for PTemp and Latitude--##
num_bins_lat = 10
num_bins_ptemp = 10

##--Define function that creates datasets from filenames--##
def find_files(directory, flight):
    ##--flight data are stored in a folder called "raw"--##
    flight_dir = os.path.join(directory, "raw", flight)
    search_pattern = os.path.join(flight_dir, "*.ict")
    return sorted(glob.glob(search_pattern))

all_dfs = []

##--Loop through each flight in the list--##
for flight in flights_to_analyze:
    
    #########################
    ##--Open ICARTT Files--##
    #########################

    dataset = icartt.Dataset(find_files(directory, flight)[0])

    ####################################
    ##--Assign date to flight number--##
    ####################################
    
    if flight=="Flight2":
        flight_date = date(2018, 4, 27)
    elif flight=="Flight10":
        flight_date = date(2018, 4, 17)
    elif flight=="Flight11": 
        flight_date = date(2018, 5, 18)
    elif flight=="Flight12":
        flight_date = date(2018, 5, 19)
    
    #################
    ##--Pull data--##
    #################
    
    ##--Ambient Data--##
    altitude = dataset.data['G_ALT'] # in m 
    latitude = dataset.data['G_LAT'] # in degrees
    temperature = dataset.data['T'] # in K
    pressure = dataset.data['P'] * 100 # in Pa
    
    ##--Hydrogen oxides from ATHOS--##
    OH = dataset.data['OH_ATHOS'] # pptv
    HO2 = dataset.data['HO2_ATHOS'] # pptv
    
    ##--CIT-CIMS (CF3O-) for peroxides and acids--##
    H2O2 = dataset.data['H2O2_CIT'] # pptv
    HNO3 = dataset.data['HNO3_CIT'] # pptv
    SO2 = dataset.data['SO2_CIT'] # pptv
    
    ##--possibly include GMI model results--##
    
    ##--NOAA Picarro--##
    CO = dataset.data['CO_NOAA'] # ppb
    CO2 = dataset.data['CO2_NOAA'] # ppm
    CH4 = dataset.data['CH4_NOAA'] # ppb
    
    CO_CO2 = CO / CO2 # ppb/ppm
    
    ##--NOAA I-CIMS--##
    #HCOOH = dataset.data['HCOOH_NOAACIMS'] # ppt
    #N2O5 = dataset.data['N2O5_ppt_NOAACIMS'] # ppt
    
    ##--NCAR chemiluminescence instrument--##
    O3 = dataset.data['O3_CL'] # ppbv
    NO = dataset.data['NO_CL'] # ppbv
    NO2 = dataset.data['NO2_CL'] # ppbv
    NOy = dataset.data['NOy_CL'] # ppbv
    
    #######################################
    ##--Calculate potential temperature--##
    #######################################
    
    ##--Constants--##
    p_0 = 1E5 # Reference pressure in Pa (1000 hPa)
    k = 0.286 # Poisson constant for dry air
    
    ##--Generate empty list for potential temperature output--##
    potential_temp = []
    
    ##--Calculate potential temperature from ambient temp & pressure--##
    for T, P in zip(temperature, pressure):
        p_t = T*(p_0/P)**k
        potential_temp.append(p_t)
    
    ##--Place in dataframe--##
    
    df = pd.DataFrame({'PTemp': potential_temp, 'latitude': latitude, 'OH':OH, 
                       'O3': O3, 'HO2': HO2, 'H2O2': H2O2, 'CO': CO, 'CO2': CO2,
                       'CO_CO2': CO_CO2, 'CH4': CH4, 'NOy': NOy, 
                       'NO': NO, 'NO2': NO2, 'SO2': SO2, 'HNO3': HNO3})
    
    ##--Drop rows where the potential temperature is above 310 K for comparison to other campaigns--##
    df = df[df['PTemp']<310]
    
    ##--And where the latitude is outside of the Arctic--##
    df = df[df['latitude']>66.5]
    
    all_dfs.append(df)
    
###########################
##--Create 2D histogram--##
###########################

##--Create GLOBAL bin edges for consistency--##
global_lat_edges = np.linspace(66.5, 86.5, num_bins_lat + 1)
global_ptemp_edges = np.linspace(235, 310, num_bins_ptemp + 1)

##--Define a function for calculating 2d MEDIAN values--##
def compute_2d_median(df, value_col, lat_edges, ptemp_edges):

    ##--Concatenate values from all input dataframes--##
    all_lat = df['latitude'].values
    all_ptemp = df['PTemp'].values
    all_val = df[value_col].values 

    ##--Create a mask of the nan values--##
    nanmask = (~np.isnan(all_lat) & ~np.isnan(all_ptemp) & ~np.isnan(all_val))
    
    ##--Handle empty plots with no data--##
    if nanmask.sum() == 0:
     ##--Create the plot using the edges and fill with NaNs--##
     return np.full((len(lat_edges)-1, len(ptemp_edges)-1), np.nan)
 
    ##--Filter out zero values, too--##
    valid = all_val > 0

    ##--Pull the median stat per bin, ignoring all other outputs--##
    stat, _, _, _ = binned_statistic_2d(all_lat[nanmask & valid], all_ptemp[nanmask & valid],
        all_val[nanmask & valid], statistic="median", bins=[lat_edges, ptemp_edges])
    
    ##--Count per bin--##
    counts, _, _, _ = binned_statistic_2d(
        all_lat[nanmask & valid], all_ptemp[nanmask & valid], all_val[nanmask & valid],
        statistic="count", bins=[lat_edges, ptemp_edges])
    
    min_count = 3

    ##--Mask bins with fewer than min_count datapoints--##
    stat[counts < min_count] = np.nan
    
    ##--Function returns binned median statistic--##
    return stat

gas_var_list = ['OH', 'HO2', 'H2O2', 'O3', 'CO', 'CO2', 'CO_CO2', 'CH4', 
                'NOy', 'NO', 'NO2', 'SO2', 'HNO3']


binned_medians = {}

for var in gas_var_list: 
    ##--Run function to get binned median nucleation data--##
    binned_medians[var] = compute_2d_median(df, var, 
                                global_lat_edges, global_ptemp_edges)


################
##--PLOTTING--##
################

##--Set number of bins for latitude and potential temperature--##
num_bins_lat = 12
num_bins_ptemp = 12
 
def plot_curtain(bin_medians, x_edges, y_edges, title, cbar_label):
    fig, ax = plt.subplots(figsize=(6, 6))
 
    ##--Makecolor map where 0 values are white--##
    new_cmap = plt.get_cmap('viridis')
    new_cmap.set_under('w')
 
    ##--Plot the 2D data using pcolormesh--##
    mesh = ax.pcolormesh(x_edges, y_edges, bin_medians.T, shading="auto", cmap=new_cmap)
 
    ##--Add colorbar--##
    cb = fig.colorbar(mesh, ax=ax, orientation='horizontal', location='bottom', pad=0.15) 
    cb.minorticks_on()
    cb.ax.tick_params(labelsize=18)
    cb.set_label(cbar_label, fontsize=18)

    ##--Set axis labels and title--##
    ax.set_xlabel("Latitude (°)", fontsize=18)
    ax.set_ylabel("Potential Temperature \u0398 (K)", fontsize=18)
    ax.tick_params(axis='both', labelsize=18)
    ax.set_title(title, fontsize=20)
    #ax.set_ylim(238, 316)
    #ax.set_xlim(64, 86)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))

    plt.tight_layout()
    plt.show()
    
##--OH--##
plot_curtain(binned_medians['OH'], global_lat_edges, global_ptemp_edges, 
             title="OH", cbar_label="OH (pptv)")
 
##--HO2--##
plot_curtain(binned_medians['HO2'], global_lat_edges, global_ptemp_edges,
             title='HO2', cbar_label = "HO2 (pptv)")

##--H2O2--##
plot_curtain(binned_medians['H2O2'], global_lat_edges, global_ptemp_edges,
             title='H2O2', cbar_label = "H2O2 (pptv)")

##--O3--##
plot_curtain(binned_medians['O3'], global_lat_edges, global_ptemp_edges,
             title='O3', cbar_label = "O3 (ppbv)")

##--CO--##
plot_curtain(binned_medians['CO'], global_lat_edges, global_ptemp_edges,
             title='CO', cbar_label = "CO (ppbv)")

##--CO2--##
plot_curtain(binned_medians['CO2'], global_lat_edges, global_ptemp_edges,
             title='CO2', cbar_label = "CO2 (ppmv)")

##--CO/CO2--##
plot_curtain(binned_medians['CO_CO2'], global_lat_edges, global_ptemp_edges,
             title='CO/CO2 ratio', cbar_label = "CO/CO2 (ppbv/ppmv)")

##--CH4--##
plot_curtain(binned_medians['CH4'], global_lat_edges, global_ptemp_edges,
             title='CH4', cbar_label = "CH4 (ppbv)")

##--NOy--##
plot_curtain(binned_medians['NOy'], global_lat_edges, global_ptemp_edges,
             title='NOy', cbar_label = "NOy (ppbv)")

##--NO--##
plot_curtain(binned_medians['NO'], global_lat_edges, global_ptemp_edges,
             title='NO', cbar_label = "NO (ppbv)")

##--NO2--##
plot_curtain(binned_medians['NO2'], global_lat_edges, global_ptemp_edges,
             title='NO2', cbar_label = "NO2 (ppbv)")

##--HNO3--##
plot_curtain(binned_medians['HNO3'], global_lat_edges, global_ptemp_edges,
             title='HNO3', cbar_label = "HNO3 (pptv)")

##--SO2--##
plot_curtain(binned_medians['SO2'], global_lat_edges, global_ptemp_edges,
             title='SO2', cbar_label = "SO2 (pptv)")
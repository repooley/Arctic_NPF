# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 13:43:39 2025

@author: repooley
"""

import icartt
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from scipy.stats import binned_statistic_2d
 
###################
##--User inputs--##
###################
 
##--Set the base directory to project folder--##
directory = r"C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data\raw"

##--Choose which flights to analyze here!--##
flights_to_analyze = ["Flight2", "Flight3", "Flight4", "Flight5", "Flight6", "Flight7", "Flight8", "Flight9", "Flight10"]

##--Define number of bins here--##
num_bins_lat = 10
num_bins_ptemp = 10

#########################
##--Open ICARTT Files--##
#########################
 
##--Define a function to find all flight data--##
def get_all_flights(directory):
    ##--flights are iteratively named Flight1, Flight2, etc--##
    raw_dir = os.path.join(directory)
    return [flight for flight in os.listdir(raw_dir) if 
            os.path.isdir(os.path.join(raw_dir, flight)) and flight.startswith("Flight")]
 
##--Define a function that creates datasets from filenames--##
def find_files(flight_dir, partial_name):
    search_pattern = os.path.join(flight_dir, f"*{partial_name}*")
    return sorted(glob.glob(search_pattern))

##--Store processed data here: --##
O3_dfs = []
CO_dfs = []
CO2_dfs = []
 
##--Loop through each flight, pulling and analyzing data--##
for flight in flights_to_analyze:
    ##--Follow which flight is processing--##
    print(f"Processing {flight}...")
    ##--Populate flight_dir established in above function--##
    flight_dir = os.path.join(directory, flight)
    ##--Pull meteorological data from AIMMS monitoring system--##
    aimms_files = find_files(flight_dir, "AIMMS_POLAR6")
    if aimms_files:
        aimms = icartt.Dataset(aimms_files[0])
    else:
        print(f"No AIMMS_POLAR6 file found for {flight}. Skipping...")
        continue  # Skip to the next flight if AIMMS file is missing
 
    ##--Trace Gas files--##
    H2O_files = find_files(flight_dir, 'H2O')

    if H2O_files:
        H2O = icartt.Dataset(H2O_files[0])
    else:
        print(f"Missing H2O data for {flight}. Skipping...")
        continue

    CO_files = find_files(flight_dir, "CO_POLAR6")
    CO2_files = find_files(flight_dir, "CO2_POLAR6")
    
    if CO_files and CO2_files:
        CO = icartt.Dataset(CO_files[0])
        CO2 = icartt.Dataset(CO2_files[0])
    else: 
        print(f"Missing CO or CO2 data for {flight}. Skipping...")
        continue
        
    ##--Flight 2 has multiple ozone files requiring special handling--##
    O3_files = find_files(flight_dir, "O3_")
    
    if len(O3_files) == 0:
        raise FileNotFoundError("No O3 files found.")
        
    elif len(O3_files) == 1 or flight != "Flight2": 
        O3 = icartt.Dataset(O3_files[0])
        O3_2 = None
    ##--Special case for Flight 2--##
    else: 
        O3 = icartt.Dataset(O3_files[0])
        O3_2 = icartt.Dataset(O3_files[1])
        
    #########################
    ##--Pull & align data--##
    #########################

    ##--AIMMS Data--##
    altitude = aimms.data['Alt'] # in m
    latitude = aimms.data['Lat'] # in degrees
    aimms_time =aimms.data['TimeWave'] # in seconds since midnight
    temperature = aimms.data['Temp']
    pressure = aimms.data['BP'] #in pa

    ##--O3 data--##
    ##--Put O3 data in list to make concatenation easier--##
    O3_starttime = list(O3.data['Start_UTC'])
    O3_conc = list(O3.data['O3'])

    ##--Check for O3_2 data and stich to end of O3 if populated--##
    if O3_2 is not None:
        O3_starttime += list(O3_2.data['Start_UTC'])
        O3_conc += list(O3_2.data['O3'])

    ##--Arbitary reference date for datetime conversion--##
    reference_date = pd.to_datetime('2015-01-01')

    ##--O3 data: addressing different data resolution compared to AIMMS--##
    ##--Convert O3_starttime to a datetime object--##
    O3_starttime_dt = pd.to_datetime(O3_starttime, unit='s', origin=reference_date)

    ##--Calculate the seconds since midnight--##
    O3_seconds_since_midnight = O3_starttime_dt.hour * 3600 + O3_starttime_dt.minute * 60 + O3_starttime_dt.second

    ##--Create O3 dataframe--##
    O3_df = pd.DataFrame({'Time_UTC': O3_seconds_since_midnight,'O3': O3_conc})

    ##--Reindex O3 data to AIMMS time and set non-overlapping time values to NaN--##
    O3_aligned = O3_df.set_index('Time_UTC').reindex(aimms_time)
    O3_aligned['O3'] = O3_aligned['O3'].where(O3_aligned.index.isin(aimms_time), np.nan)
    O3_conc_aligned = O3_aligned['O3']

    ##--CO and CO2--##
    CO_conc = CO.data['CO_ppbv']
    CO_time = CO.data['Time_UTC']
    CO2_conc = CO2.data['CO2_ppmv']
    CO2_time = CO2.data['Time_UTC']

    CO_df = pd.DataFrame({'time': CO_time, 'conc': CO_conc}).set_index('time')
    CO_conc_aligned = CO_df.reindex(aimms_time)['conc']

    CO2_df = pd.DataFrame({'time':CO2_time, 'conc': CO2_conc}).set_index('time')
    CO2_conc_aligned = CO2_df.reindex(aimms_time)['conc']
    
    ####################
    ##--Calculations--##
    ####################

    ##--Convert absolute temperature to potential temperature--##
    ##--Constants--##
    p_0 = 1E5 # Reference pressure in Pa (1000 hPa)
    k = 0.286 # Poisson constant for dry air

    ##--Convert temperature from Celcius to Kelvin--##
    temperature_k = np.array(temperature) + 273.15

    ##--Generate empty list for potential temperature output--##
    potential_temp = []

    ##--Calculate potential temperature from ambient temp & pressure--##
    for T, P in zip(temperature_k, pressure):
        p_t = T*(p_0/P)**k
        potential_temp.append(p_t)
        
    #########################
    ##--Create dataframes--##
    #########################
    
    ##--Creates separate dfs to preserve data--##
    O3_df = pd.DataFrame({'PTemp': potential_temp, 'Latitude': latitude, 'O3_conc': O3_conc_aligned}).dropna()
    CO_df = pd.DataFrame({'PTemp': potential_temp, 'Latitude': latitude, 'CO_conc': CO_conc_aligned}).dropna()
    CO2_df = pd.DataFrame({'PTemp': potential_temp, 'Latitude': latitude, 'CO2_conc': CO2_conc_aligned}).dropna()

    ##--Store all processed data and ensure in numpy arrays--##
    O3_dfs.append(O3_df[['PTemp', 'Latitude', 'O3_conc']])
    CO_dfs.append(CO_df[['PTemp', 'Latitude', 'CO_conc']])
    CO2_dfs.append(CO2_df[['PTemp', 'Latitude', 'CO2_conc']])

###########################
##--Prepare for Binning--##
###########################
 
##--Binning for O3 data--##
all_latitudes_O3 = np.concatenate([df["Latitude"].values for df in O3_dfs])
all_ptemps_O3 = np.concatenate([df["PTemp"].values for df in O3_dfs])
all_O3 = np.concatenate([df["O3_conc"].values for df in O3_dfs])
 
lat_bin_edges_O3 = np.linspace(all_latitudes_O3.min(), all_latitudes_O3.max(), num_bins_lat + 1)
ptemp_bin_edges_O3 = np.linspace(all_ptemps_O3.min(), all_ptemps_O3.max(), num_bins_ptemp + 1)
 
O3_bin_medians, _, _, _ = binned_statistic_2d(all_latitudes_O3, all_ptemps_O3, 
        all_O3, statistic="median", bins=[lat_bin_edges_O3, ptemp_bin_edges_O3])
 
##--Binning for CO data--##
all_latitudes_CO = np.concatenate([df["Latitude"].values for df in CO_dfs])
all_ptemps_CO = np.concatenate([df["PTemp"].values for df in CO_dfs])
all_CO = np.concatenate([df["CO_conc"].values for df in CO_dfs])
 
lat_bin_edges_CO = np.linspace(all_latitudes_CO.min(), all_latitudes_CO.max(), num_bins_lat + 1)
ptemp_bin_edges_CO = np.linspace(all_ptemps_CO.min(), all_ptemps_CO.max(), num_bins_ptemp + 1)
 
CO_bin_medians, _, _, _ = binned_statistic_2d(all_latitudes_CO, all_ptemps_CO, 
        all_CO, statistic="median", bins=[lat_bin_edges_CO, ptemp_bin_edges_CO])

##--Binning for CO2 data--##
all_latitudes_CO2 = np.concatenate([df["Latitude"].values for df in CO2_dfs])
all_ptemps_CO2 = np.concatenate([df["PTemp"].values for df in CO2_dfs])
all_CO2 = np.concatenate([df["CO2_conc"].values for df in CO2_dfs])
 
lat_bin_edges_CO2 = np.linspace(all_latitudes_CO2.min(), all_latitudes_CO2.max(), num_bins_lat + 1)
ptemp_bin_edges_CO2 = np.linspace(all_ptemps_CO2.min(), all_ptemps_CO2.max(), num_bins_ptemp + 1)
 
CO2_bin_medians, _, _, _ = binned_statistic_2d(all_latitudes_CO2, all_ptemps_CO2, 
        all_CO2, statistic="median", bins=[lat_bin_edges_CO2, ptemp_bin_edges_CO2])

################
##--PLOTTING--##
################
 
def plot_curtain(bin_medians, x_edges, y_edges, vmin, vmax, title, cbar_label, output_path):
    fig, ax = plt.subplots(figsize=(8, 6))
 
    ##--Make special color map where 0 values are white--##
    new_cmap = plt.get_cmap('cividis')
    new_cmap.set_under('w')
 
    ##--Plot the 2D data using pcolormesh--##
    mesh = ax.pcolormesh(x_edges, y_edges, bin_medians.T, shading="auto", cmap=new_cmap, vmin=vmin, vmax=vmax)
 
    ##--Add colorbar--##
    cb = fig.colorbar(mesh, ax=ax)
    cb.minorticks_on()
    cb.ax.tick_params(labelsize=16)
    cb.set_label(cbar_label, fontsize=16)
    
    ##--Add dashed horizontal lines for the polar dome boundaries--##
    ##--Boundaries are defined from Bozem et al 2019 (ACP)--##
    ax.axhline(y=285, color='k', linestyle='--', linewidth=1)
    ax.axhline(y=299, color='k', linestyle='--', linewidth=1)
    '''
    ##--Add text labels on the left-hand side within the plot area--##
    ##--Compute midpoints for label placement--##
    polar_dome_mid = (238 + 275) / 2
    marginal_polar_dome_mid = (275 + 299) / 2
    x_text = ax.get_xlim()[0] - 0.25  # left edge plus a small offset
    
    ax.text(x_text, polar_dome_mid, 'Polar Dome',
            rotation=90, fontsize=10, color='k',
            verticalalignment='center', horizontalalignment='center')
    ax.text(x_text, marginal_polar_dome_mid, 'Marginal Polar Dome',
            rotation=90, fontsize=10, color='k',
            verticalalignment='center', horizontalalignment='center')
    '''
    ##--Set axis labels and title--##
    ax.set_xlabel("Latitude (°)", fontsize=18)
    ax.set_ylabel("Potential Temperature \u0398 (K)", fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    ax.set_title(title, fontsize=18)
    #ax.set_ylim(238, 301)
    #ax.set_xlim(79.5, 83.7)
    
    ##--Save the plot--##
    #plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.tight_layout()
    plt.show()
 
##--Plot for O3--##
plot_curtain(O3_bin_medians, lat_bin_edges_O3, ptemp_bin_edges_O3, vmin=0, vmax=75,
    title="O\u2083 Mixing Ratio", cbar_label="O\u2083 ppbv",
    output_path=r"C:\Users\repooley\REP_PhD\NETCARE2015\data\processed\CurtainPlots\TraceGas\PTempLatitude\O3_MultiFlights.png")

##--Plot for CO--##
plot_curtain(CO_bin_medians, lat_bin_edges_CO, ptemp_bin_edges_CO, vmin=110, vmax=155,
    title="CO Mixing Ratio", cbar_label="CO ppbv",
    output_path=r"C:\Users\repooley\REP_PhD\NETCARE2015\data\processed\CurtainPlots\TraceGas\PTempLatitude\CO_MultiFlights.png")

##--Plot for RH wrt Ice--##
plot_curtain(CO2_bin_medians, lat_bin_edges_CO2, ptemp_bin_edges_CO2, vmin=400, vmax=410,
    title="CO\u2082 Mixing Ratio", cbar_label="CO\u2082 ppmv",
    output_path=r"C:\Users\repooley\REP_PhD\NETCARE2015\data\processed\CurtainPlots\TraceGas\PTempLatitude\CO2_MultiFlights.png")

########################
##--Diagnostic Plots--##
########################

##--Remove hashtags below to comment out this section--##
#'''

##--Counts per bin for O3 data--##
O3_bin_counts, _, _, _ = binned_statistic_2d(all_latitudes_O3, all_ptemps_O3, all_O3,
    statistic="count", bins=[lat_bin_edges_O3, ptemp_bin_edges_O3])
 
##--Counts per bin for CPC10 data--##
CO_bin_counts, _, _, _ = binned_statistic_2d(all_latitudes_CO, all_ptemps_CO, all_CO,
    statistic="count", bins=[lat_bin_edges_CO, ptemp_bin_edges_CO])
 
##--Counts per bin for N3-10 particles--##
CO2_bin_counts, _, _, _ = binned_statistic_2d(all_latitudes_CO2, all_ptemps_CO2, all_CO2,
    statistic="count", bins=[lat_bin_edges_CO2, ptemp_bin_edges_CO2])

##--Plotting--##

def plot_curtain(bin_counts, x_edges, y_edges, vmin, vmax, title, cbar_label, output_path):
    fig, ax = plt.subplots(figsize=(8, 6))
 
    ##--Set NaN values to white--##
    cmap = plt.get_cmap('inferno')
    cmap.set_under('w')
 
    ##--Plot the 2D data using pcolormesh--##
    mesh = ax.pcolormesh(x_edges, y_edges, bin_counts.T, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
 
    ##--Add colorbar--##
    cb = fig.colorbar(mesh, ax=ax)
    cb.minorticks_on()
    cb.set_label(cbar_label, fontsize=12)
    
    ##--Add dashed horizontal lines for the polar dome boundaries--##
    ax.axhline(y=275, color='k', linestyle='--', linewidth=1)
    ax.axhline(y=299, color='k', linestyle='--', linewidth=1)
    
    '''
    ##--Add labels on the left-hand side within the plot area--##
    polar_dome_mid = (238 + 275) / 2
    marginal_polar_dome_mid = (275 + 299) / 2
    x_text = ax.get_xlim()[0] - 0.25 
    
    ax.text(x_text, polar_dome_mid, 'Polar Dome',
            rotation=90, fontsize=10, color='k',
            verticalalignment='center', horizontalalignment='center')
    ax.text(x_text, marginal_polar_dome_mid, 'Marginal Polar Dome',
            rotation=90, fontsize=10, color='k',
            verticalalignment='center', horizontalalignment='center')
    '''
    ##--Set axis labels and title--##
    ax.set_xlabel("Latitude (°)", fontsize=12)
    ax.set_ylabel("Potential Temperature Θ (K)", fontsize=12)
    ax.set_title(title)
    #ax.set_ylim(238, 301)
    #ax.set_xlim(79.5, 83.7)
    
    ##--Save the plot--##
    #plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.tight_layout()
    plt.show()
 
##--Plot for O3 counts--##
plot_curtain(O3_bin_counts, lat_bin_edges_O3, ptemp_bin_edges_O3, vmin=1, vmax=500, 
    title="O\u2083 Data Point Counts", cbar_label="Number of Data Points",
    output_path=r"C:\Users\repooley\REP_PhD\NETCARE2015\data\processed\CurtainPlots\TraceGas\PTempLatitude\O3_MultiFlights_diagnostic.png")
 
##--Plot for CPC10 counts--##
plot_curtain(CO_bin_counts, lat_bin_edges_CO, ptemp_bin_edges_CO, vmin=1, vmax=6000,  
    title="CO Data Point Counts", cbar_label="Number of Data Points",
    output_path=r"C:\Users\repooley\REP_PhD\NETCARE2015\data\processed\CurtainPlots\TraceGas\PTempLatitude\CO_MultiFlights_diagnostic.png")
 
##--Plot for CO2 counts--##
plot_curtain(CO2_bin_counts, lat_bin_edges_CO2, ptemp_bin_edges_CO2, vmin=1, vmax=6000,  
    title="CO\u2082 Data Point Counts", cbar_label="Number of Data Points",
    output_path=r"C:\Users\repooley\REP_PhD\NETCARE2015\data\processed\CurtainPlots\Nucleating\PTempLatitude\CO2_MultiFlights_diagnostic.png")

#'''
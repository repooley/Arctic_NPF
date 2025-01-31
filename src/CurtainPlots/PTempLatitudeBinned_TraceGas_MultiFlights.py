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
 
#########################
##--Open ICARTT Files--##
#########################
 
##--Set the base directory to project folder--##
directory = r"C:\Users\repooley\REP_PhD\NETCARE2015\data\raw"
 
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
 
##--Choose which flights to analyze here!--##
##--Flight1 AIMMS file currently broken, Flight8 has very high O3, CO--##
##--Flights 9 and 10 are in a different region, plot separately--##
flights_to_analyze = ["Flight2", "Flight3", "Flight4", "Flight5", "Flight6", "Flight7"]

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
        
    #################
    ##--Pull data--##
    #################

    ##--AIMMS Data--##
    altitude = aimms.data['Alt'] # in m
    latitude = aimms.data['Lat'] # in degrees
    aimms_time =aimms.data['TimeWave'] # in seconds since midnight
    temperature = aimms.data['Temp'] #in C
    pressure = aimms.data['BP'] #in pa

    ##--Trace Gas Data--##
    CO_conc = CO.data['CO_ppbv']
    CO2_conc = CO2.data['CO2_ppmv']

    ##--Put O3 data in list to make concatenation easier--##
    O3_starttime = list(O3.data['Start_UTC'])
    O3_conc = list(O3.data['O3'])

    ##--Check for O3_2 data and stich to end of O3 if populated--##
    if O3_2 is not None:
        O3_starttime += list(O3_2.data['Start_UTC'])
        O3_conc += list(O3_2.data['O3'])
        
    ##################
    ##--Align time--##
    ##################

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

    ##--Other trace gas data: addressing different start/stop times than AIMMS--##
    aimms_start = aimms_time.min()
    aimms_end = aimms_time.max()

    ##--Handle CO data with different start/stop times than AIMMS--##
    CO_time = CO.data['Time_UTC']

    ##--Trim CO data if it starts before AIMMS--##
    if CO_time.min() < aimms_start:
        mask_start = CO_time >= aimms_start
        CO_time = CO_time[mask_start]
        CO_conc = CO_conc[mask_start]
        
    ##--Append CO data with NaNs if it ends before AIMMS--##
    if CO_time.max() < aimms_end: 
        missing_times = np.arange(CO_time.max()+1, aimms_end +1)
        CO_time = np.concatenate([CO_time, missing_times])
        CO_conc = np.concatenate([CO_conc, [np.nan]*len(missing_times)])

    ##--Create a DataFrame for CO data and reindex to AIMMS time, setting non-overlapping times to nan--##
    CO_df = pd.DataFrame({'Time_UTC': CO_time, 'CO_ppbv': CO_conc})
    CO_aligned = CO_df.set_index('Time_UTC').reindex(aimms_time)
    CO_aligned['CO_ppbv']= CO_aligned['CO_ppbv'].where(CO_aligned.index.isin(aimms_time), np.nan)
    CO_conc_aligned = CO_aligned['CO_ppbv']

    ##--Handle CO2 data with different start/stop times than AIMMS--##
    CO2_time = CO2.data['Time_UTC']

    ##--Trim CO2 data if it starts before AIMMS--##
    if CO2_time.min() < aimms_start:
        mask_start = CO2_time >= aimms_start
        CO2_time = CO2_time[mask_start]
        CO2_conc = CO2_conc[mask_start]
        
    ##--Append CO2 data with NaNs if it ends before AIMMS--##
    if CO2_time.max() < aimms_end: 
        missing_times = np.arange(CO2_time.max()+1, aimms_end +1)
        CO2_time = np.concatenate([CO2_time, missing_times])
        CO2_conc = np.concatenate([CO2_conc, [np.nan]*len(missing_times)])

    ##--Create a DataFrame for CO2 data and reindex to AIMMS time, setting non-overlapping times to nan--##
    CO2_df = pd.DataFrame({'Time_UTC': CO2_time, 'CO2_ppmv': CO2_conc})
    CO2_aligned = CO2_df.set_index('Time_UTC').reindex(aimms_time)
    CO2_aligned['CO2_ppmv']=CO2_aligned['CO2_ppmv'].where(CO2_aligned.index.isin(aimms_time), np.nan)
    CO2_conc_aligned = CO2_aligned['CO2_ppmv']
    
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
 
##--Define number of bins here--##
num_bins = 30
 
##--Binning for O3 data--##
all_latitudes_O3 = np.concatenate([df["Latitude"].values for df in O3_dfs])
all_ptemps_O3 = np.concatenate([df["PTemp"].values for df in O3_dfs])
all_O3 = np.concatenate([df["O3_conc"].values for df in O3_dfs])
 
lat_bin_edges_O3 = np.linspace(all_latitudes_O3.min(), all_latitudes_O3.max(), num_bins + 1)
ptemp_bin_edges_O3 = np.linspace(all_ptemps_O3.min(), all_ptemps_O3.max(), num_bins + 1)
 
O3_bin_medians, _, _, _ = binned_statistic_2d(all_latitudes_O3, all_ptemps_O3, 
        all_O3, statistic="mean", bins=[lat_bin_edges_O3, ptemp_bin_edges_O3])
 
##--Binning for CO data--##
all_latitudes_CO = np.concatenate([df["Latitude"].values for df in CO_dfs])
all_ptemps_CO = np.concatenate([df["PTemp"].values for df in CO_dfs])
all_CO = np.concatenate([df["CO_conc"].values for df in CO_dfs])
 
lat_bin_edges_CO = np.linspace(all_latitudes_CO.min(), all_latitudes_CO.max(), num_bins + 1)
ptemp_bin_edges_CO = np.linspace(all_ptemps_CO.min(), all_ptemps_CO.max(), num_bins + 1)
 
CO_bin_medians, _, _, _ = binned_statistic_2d(all_latitudes_CO, all_ptemps_CO, 
        all_CO, statistic="mean", bins=[lat_bin_edges_CO, ptemp_bin_edges_CO])

##--Binning for CO2 data--##
all_latitudes_CO2 = np.concatenate([df["Latitude"].values for df in CO2_dfs])
all_ptemps_CO2 = np.concatenate([df["PTemp"].values for df in CO2_dfs])
all_CO2 = np.concatenate([df["CO2_conc"].values for df in CO2_dfs])
 
lat_bin_edges_CO2 = np.linspace(all_latitudes_CO2.min(), all_latitudes_CO2.max(), num_bins + 1)
ptemp_bin_edges_CO2 = np.linspace(all_ptemps_CO2.min(), all_ptemps_CO2.max(), num_bins + 1)
 
CO2_bin_medians, _, _, _ = binned_statistic_2d(all_latitudes_CO2, all_ptemps_CO2, 
        all_CO2, statistic="mean", bins=[lat_bin_edges_CO2, ptemp_bin_edges_CO2])

################
##--PLOTTING--##
################
 
def plot_curtain(bin_medians, x_edges, y_edges, vmin, vmax, title, cbar_label, output_path):
    fig, ax = plt.subplots(figsize=(8, 6))
 
    ##--Make special color map where 0 values are white--##
    new_cmap = plt.get_cmap('viridis')
    new_cmap.set_under('w')
 
    ##--Plot the 2D data using pcolormesh--##
    mesh = ax.pcolormesh(x_edges, y_edges, bin_medians.T, shading="auto", cmap=new_cmap, vmin=vmin, vmax=vmax)
 
    ##--Add colorbar--##
    cb = fig.colorbar(mesh, ax=ax)
    cb.minorticks_on()
    cb.set_label(cbar_label, fontsize=12)
    
    ##--Add dashed horizontal lines for the polar dome boundaries--##
    ##--Boundaries are defined from Bozem et al 2019 (ACP)--##
    ax.axhline(y=275, color='k', linestyle='--', linewidth=1)
    ax.axhline(y=299, color='k', linestyle='--', linewidth=1)
    
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
 
    ##--Set axis labels and title--##
    ax.set_xlabel("Latitude (°)", fontsize=12)
    ax.set_ylabel("Potential Temperature \u0398 (K)", fontsize=12)
    ax.set_title(title)
    ax.set_ylim(238, 301)
    ax.set_xlim(79.5, 83.7)
 
    ##--Save the plot--##
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.tight_layout()
    plt.show()
 
##--Plot for O3--##
plot_curtain(O3_bin_medians, lat_bin_edges_O3, ptemp_bin_edges_O3, vmin=0, vmax=75,
    title="O\u2083 Mixing Ratio", cbar_label="O\u2083 ppbv",
    output_path=r"C:\Users\repooley\REP_PhD\NETCARE2015\data\processed\CurtainPlots\TraceGas\PTempLatitude\O3_MultiFlights.png")

##--Plot for CO--##
plot_curtain(CO_bin_medians, lat_bin_edges_CO, ptemp_bin_edges_CO, vmin=110, vmax=175,
    title="CO Mixing Ratio", cbar_label="CO ppbv",
    output_path=r"C:\Users\repooley\REP_PhD\NETCARE2015\data\processed\CurtainPlots\TraceGas\PTempLatitude\CO_MultiFlights.png")

##--Plot for RH wrt Ice--##
plot_curtain(CO2_bin_medians, lat_bin_edges_CO2, ptemp_bin_edges_CO2, vmin=400, vmax=410,
    title="CO\u2082 Mixing Ratio", cbar_label="CO\u2082 ppmv",
    output_path=r"C:\Users\repooley\REP_PhD\NETCARE2015\data\processed\CurtainPlots\TraceGas\PTempLatitude\CO2_MultiFlights.png")
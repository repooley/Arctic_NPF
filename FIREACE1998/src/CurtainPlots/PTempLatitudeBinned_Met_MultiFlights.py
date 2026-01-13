# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 10:25:54 2025

@author: repooley
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
from scipy.stats import binned_statistic_2d
import cmcrameri as cm 

###################
##--User inputs--##
###################

##--Set the base directory to project folder--##
directory = r"C:\Users\repooley\REP_PhD\Arctic_NPF\FIREACE1998\data\raw"

##--Choose which flights to analyze here!--##
flights_to_analyze = ["Flight3", 
                      "Flight7", "Flight8", "Flight9", "Flight10", "Flight11", "Flight12",
                      "Flight13", "Flight14", "Flight15", "Flight16", "Flight17", "Flight18"]

##--Set binning for PTemp and Latitude--##
##--Define number of bins here--##
num_bins_lat = 4
num_bins_ptemp = 8

##--Separate bin numbers for the averaged data--##
num_bins_lat_averaged = 6
num_bins_ptemp_averaged = 6

##--Base output path in directory--##
output_path = r"C:\Users\repooley\REP_PhD\Arctic_NPF\FIREACE1998\data\processed"
 
##################
##--Open Files--##
##################

##--Define function that creates datasets from filenames--##
def find_files(directory, flight, partial_name):
    ##--flight data are stored in a folder called "raw"--##
    flight_dir = os.path.join(directory, flight)
    search_pattern = os.path.join(flight_dir, f"*{partial_name}*")
    return sorted(glob.glob(search_pattern))

##--Define a function to find all flight data--##
def get_all_flights(directory):
    ##--flights are iteratively named Flight1, Flight2, etc--##
    raw_dir = os.path.join(directory)
    return [flight for flight in os.listdir(raw_dir) if 
            os.path.isdir(os.path.join(raw_dir, flight)) and flight.startswith("Flight")]

#%%
#################
##--Pull data--##
#################

##--Store processed data here: --##
temp_dfs = []
RH_dfs = []
 
##--Loop through each flight, pulling and analyzing data--##
for flight in flights_to_analyze:
    ##--Follow which flight is processing--##
    print(f"Processing {flight}...")

    ##--Pull csv file containing all data--##
    files = find_files(directory, flight, "FIREACE")
    
    ##--The 1 hz data is always the first file--##
    if files:
        data = pd.read_csv(files[0])
        
        ##--Replace nan values--##
        data.replace(-88.88888, np.nan, inplace=True)
        
    else:
        print(f"No FIRE-ACE file found for {flight}. Skipping...")
        continue  # Skip to the next flight if FIRE-ACE file is missing
 
    ##--Pull data variables from file--##
    time = data['Time'] # HHMMSS UTC time
    pressure = data['Pressure'] * 100 # in Pa
    temperature = data['Temperature'] + 273.15 # in K
    RH_probe = data['RH'] # percent wrt water
    altitude = data['Altitude'] # in m (agl?)
    latitude = data['Latitude'] # degrees
    print(min(latitude))
    longitude = data['Longitude'] # degrees

    #######################################
    ##--Calculate potential temperature--##
    #######################################

    ##--Convert absolute temperature to potential temperature--##
    ##--Constants--##
    p_0 = 1E5 # Reference pressure in Pa (1000 hPa)
    k = 0.286 # Poisson constant for dry air


    ##--Generate empty list for potential temperature output--##
    potential_temp = []

    ##--Calculate potential temperature from ambient temp & pressure--##
    for T, P in zip(temperature, pressure):
        p_t = T*(p_0/P)**k
        potential_temp.append(p_t)
        
    temp_df = pd.DataFrame({'temp': temperature, 'ptemp': potential_temp, 
                            'lat': latitude, 'alt': altitude})
    RH_df = pd.DataFrame({'RH': RH_probe, 'ptemp': potential_temp, 
                            'lat': latitude, 'alt': altitude})
    
    temp_dfs.append(temp_df)
    RH_dfs.append(RH_df)

###########################
##--Prepare for Binning--##
###########################

##--Compute global min/max for all flights (non-averaged) --##
all_lats = np.concatenate([df["lat"].values for df in temp_dfs])
all_ptemps = np.concatenate([df["ptemp"].values for df in temp_dfs])

lat_min, lat_max = np.nanmin(all_lats), np.nanmax(all_lats)
ptemp_min, ptemp_max = np.nanmin(all_ptemps), np.nanmax(all_ptemps)

##--Common binning edges--##
common_lat_bin_edges = np.linspace(lat_min, lat_max, num_bins_lat + 1)
common_ptemp_bin_edges = np.linspace(ptemp_min, ptemp_max, num_bins_ptemp + 1)


##--Binning for temp data--##
all_latitudes_temp = np.concatenate([df["lat"].values for df in temp_dfs])
all_ptemps_temp = np.concatenate([df["ptemp"].values for df in temp_dfs])
all_temp = np.concatenate([df["temp"].values for df in temp_dfs])

temp_bin_medians, _, _, _ = binned_statistic_2d(
    all_latitudes_temp, all_ptemps_temp, all_temp,
    statistic="median", bins=[common_lat_bin_edges, common_ptemp_bin_edges])

##--Binning for RH data--##
all_latitudes_RH = np.concatenate([df["lat"].values for df in RH_dfs])
all_ptemps_RH = np.concatenate([df["ptemp"].values for df in RH_dfs])
all_RH = np.concatenate([df["RH"].values for df in RH_dfs])

probe_RH_medians, _, _, _ = binned_statistic_2d(
    all_latitudes_RH, all_ptemps_RH, all_RH,
    statistic="median", bins=[common_lat_bin_edges, common_ptemp_bin_edges])

cmap = cm.cm.oslo

################
##--PLOTTING--##
################
 
def plot_curtain(bin_medians, x_edges, y_edges, vmin, vmax, title, cbar_label): #, output_path):
    fig, ax = plt.subplots(figsize=(6, 6))
 
    ##--Makecolor map where 0 values are white--##
    new_cmap = cmap
    new_cmap.set_under('w')
 
    ##--Plot the 2D data using pcolormesh--##
    mesh = ax.pcolormesh(x_edges, y_edges, bin_medians.T,
                     shading="flat", cmap=new_cmap, vmin=vmin, vmax=vmax)
 
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
    ax.set_ylim(238, 316)
    ax.set_xlim(64, 86)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
 
    ##--Save the plot--##
    #plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.tight_layout()
    plt.show()
 
##--Plot for probe RH--##
plot_curtain(probe_RH_medians, common_lat_bin_edges, common_ptemp_bin_edges, vmin=1, vmax=120,
    title="Probe Relative Humidity", cbar_label="% Relative Humidity")
    #output_path=f"{output_path}\\Nucleating/PTempLatitude/MultiFlights.png")

##--Plot for temperature--##
plot_curtain(temp_bin_medians,  common_lat_bin_edges, common_ptemp_bin_edges, vmin=220, vmax=310,
    title="Absolute Temperature", cbar_label='Temperature (K)')

########################
##--Diagnostic Plots--##
########################

##--Remove hashtags below to comment out this section--##
'''
##--Counts per bin for CPC3 data--##
CPC3_bin_counts, _, _, _ = binned_statistic_2d(all_latitudes_CPC3, all_ptemps_CPC3, all_CPC3_concs,
    statistic="count", bins=[lat_bin_edges_CPC3, ptemp_bin_edges_CPC3])
 
##--Counts per bin for CPC10 data--##
CPC10_bin_counts, _, _, _ = binned_statistic_2d(all_latitudes_CPC10, all_ptemps_CPC10, all_CPC10_concs,
    statistic="count", bins=[lat_bin_edges_CPC10, ptemp_bin_edges_CPC10])

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
    cb.ax.tick_params(labelsize=16)
    cb.set_label(cbar_label, fontsize=16)
    
    ##--Add dashed horizontal lines for the polar dome boundaries--##
    ax.axhline(y=285, color='k', linestyle='--', linewidth=1)
    ax.axhline(y=299, color='k', linestyle='--', linewidth=1)
    

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

 
    ##--Set axis labels and title--##
    ax.set_xlabel("Latitude (°)", fontsize=16)
    ax.set_ylabel("Potential Temperature Θ (K)", fontsize=16)
    ax.tick_params(axis='both', labelsize=16)
    ax.set_title(title, fontsize=18)
    #ax.set_ylim(238, 301)
    #ax.set_xlim(79.5, 83.7)
 
    ##--Save the plot--##
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.tight_layout()
    plt.show()
 
##--Plot for CPC3 counts--##
plot_curtain(CPC3_bin_counts, lat_bin_edges_CPC3, ptemp_bin_edges_CPC3, vmin=1, vmax=5500, 
    title="Particles >2.5 nm Data Point Counts", cbar_label="Number of Data Points",
    output_path=f"{output_path}\\CPC3/PtempLatitude/MultiFlights_diagnostic.png")
 
##--Plot for CPC10 counts--##
plot_curtain(CPC10_bin_counts, lat_bin_edges_CPC10, ptemp_bin_edges_CPC10, vmin=1, vmax=6000,  
    title="Particles >10 nm Data Point Counts", cbar_label="Number of Data Points",
    output_path=f"{output_path}\\CPC10/PtempLatitude/MultiFlights_diagnostic.png")
'''
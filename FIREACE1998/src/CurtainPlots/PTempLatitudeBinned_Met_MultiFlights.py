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
num_bins_lat = 6
num_bins_ptemp = 12

##--Separate bin numbers for the averaged data--##
num_bins_lat_averaged = 6
num_bins_ptemp_averaged = 6

PCASP_bins_path = r"C:\Users\repooley\REP_PhD\Arctic_NPF\FIREACE1998\data\raw\FIREACE1998_PCASP_bins.csv"

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
particle_dfs = []
averaged_dfs = []
 
##--Loop through each flight, pulling and analyzing data--##
for flight in flights_to_analyze:
    ##--Follow which flight is processing--##
    print(f"Processing {flight}...")

    ##--Pull csv file containing all data--##
    files = find_files(directory, flight, "FIREACE")
    
    ##--The 1 hz data is always the first file--##
    if files:
        data = pd.read_csv(files[0])
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

###########################
##--Prepare for Binning--##
###########################

##--Creates separate dfs to preserve data--##
probe_RH_df = pd.DataFrame({'PTemp': potential_temp, 'Latitude': latitude, 'Probe_RH': RH_probe})
temp_df = pd.DataFrame({'PTemp': potential_temp, 'Latitude': latitude, 'Temperature': temperature})#, 'nuc_particles': nuc_particles})

##--Drop NaNs to prevent issues with potential_temp floats--##
clean_probe_RH_df = probe_RH_df.dropna()
clean_temp_df = temp_df.dropna()

##--Compute global min/max values across all data BEFORE dropping NaNs--##
lat_min, lat_max = np.nanmin(latitude), np.nanmax(latitude)
ptemp_min, ptemp_max = np.nanmin(potential_temp), np.nanmax(potential_temp)

##--Generate common bin edges using specified number of bins--##
common_lat_bin_edges = np.linspace(lat_min, lat_max, num_bins_lat + 1)
common_ptemp_bin_edges = np.linspace(ptemp_min, ptemp_max, num_bins_ptemp + 1)


probe_RH_medians, _, _, _ = binned_statistic_2d(
    probe_RH_df['Latitude'], probe_RH_df['PTemp'], clean_probe_RH_df['Probe_RH'], 
    statistic='median', bins=[common_lat_bin_edges, common_ptemp_bin_edges])

temp_bin_medians, _, _, _ = binned_statistic_2d(
    clean_temp_df['Latitude'], clean_temp_df['PTemp'], clean_temp_df['Temperature'], 
    statistic='median', bins=[common_lat_bin_edges, common_ptemp_bin_edges])

################
##--PLOTTING--##
################
 
def plot_curtain(bin_medians, x_edges, y_edges, vmin, vmax, title, cbar_label): #, output_path):
    fig, ax = plt.subplots(figsize=(8, 6))
 
    ##--Makecolor map where 0 values are white--##
    new_cmap = plt.get_cmap('viridis')
    new_cmap.set_under('w')
 
    ##--Plot the 2D data using pcolormesh--##
    mesh = ax.pcolormesh(x_edges, y_edges, bin_medians.T, shading="auto", cmap=new_cmap, vmin=vmin, vmax=vmax)
 
    ##--Add colorbar--##
    cb = fig.colorbar(mesh, ax=ax)
    cb.minorticks_on()
    cb.ax.tick_params(labelsize=16)
    cb.set_label(cbar_label, fontsize=16)

    ##--Set axis labels and title--##
    ax.set_xlabel("Latitude (°)", fontsize=16)
    ax.set_ylabel("Potential Temperature \u0398 (K)", fontsize=16)
    ax.tick_params(axis='both', labelsize=16)
    ax.set_title(title, fontsize=18)
    #ax.set_ylim(238, 301)
    ax.set_xlim(67, 77)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
 
    ##--Save the plot--##
    #plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.tight_layout()
    plt.show()
 
##--Plot for probe RH--##
plot_curtain(probe_RH_medians, common_lat_bin_edges, common_ptemp_bin_edges, vmin=1, vmax=105,
    title="Probe Relative Humidity", cbar_label="% Relative Humidity")
    #output_path=f"{output_path}\\Nucleating/PTempLatitude/MultiFlights.png")

##--Plot for temperature--##
plot_curtain(temp_bin_medians,  common_lat_bin_edges, common_ptemp_bin_edges, vmin=1, vmax=310,
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
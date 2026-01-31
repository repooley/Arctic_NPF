# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 10:49:55 2026

@author: repooley
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from scipy.stats import binned_statistic_2d
from datetime import date
 
###################
##--User inputs--##
###################

##--Set the base directory to project folder--##
directory = r"C:\Users\repooley\REP_PhD\Arctic_NPF\PAMARCMiP2012\data\raw"
 
##--Choose which flights to analyze here!--##
flights_to_analyze = ["Flight1", "Flight2", "Flight3", "Flight6", "Flight7", "Flight8", "Flight9"]

##--Set binning for PTemp and Latitude--##
num_bins_lat = 8
num_bins_ptemp = 8

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
def find_files(directory, flight, partial_name):
    search_pattern = os.path.join(directory, flight, f"*{partial_name}*")
    return sorted(glob.glob(search_pattern))
 
##--Store processed data here: --##
temperature_dfs = []
RH_dfs = []
 
##--Loop through each flight, pulling and analyzing data--##
for flight in flights_to_analyze:
    ##--Follow which flight is processing--##
    print(f"Processing {flight}...")
    
    ##--Pull file--##
    data = pd.read_csv(find_files(directory, flight, ".csv")[0])
    
    #################
    ##--Pull data--##
    #################
    
    ##--Data--##
    altitude = data['Altitude'] # in m
    latitude = data['Latitude'] # in degrees
    temperature = data['Temp'] + 273.15 # in K
    pressure = data['Pressure'] # in pa
    time = data['Time'] # seconds since midnight
    RH = data['RH'] # percent wrt water
    
    ##--The first datapoint in 'latitude' column is erraneous (47.12 N)--##
    latitude = latitude.where(latitude >= 50, np.nan)
 
    ####################################
    ##--Assign date to flight number--##
    ####################################
    
    if flight=="Flight1":
        flight_date = date(2012, 3, 29)
    elif flight=="Flight2":
        flight_date = date(2012, 3, 30)
    elif flight=="Flight3":
        flight_date = date(2012, 4, 2)
    elif flight=="Flight4" or flight=="Flight5":
        flight_date = date(2012, 4, 3)
    elif flight=="Flight6":
        flight_date = date(2012, 4, 4)
    elif flight=="Flight7": 
        flight_date = date(2012, 4, 5)
    elif flight=="Flight8": 
        flight_date = date(2012, 4, 6)
    elif flight=="Flight9": 
        flight_date = date(2012, 4, 7)
        
    ###########################
    ##--Calc potential temp--##
    ###########################
    
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
    
    #########################
    ##--Create dataframes--##
    #########################
 
    ##--Drop NaNs, done for individual datasets for data preservation--##
    temperature_df = pd.DataFrame({'Ptemp': potential_temp, 'Latitude': latitude, 
                            'Temperature': temperature}).dropna()
    ##--Calling n 10-60 'growth'--##
    RH_df = pd.DataFrame({'Ptemp': potential_temp, 'Latitude': latitude, 
                           'RH': RH}).dropna()
    
    ##--Store all processed data and ensure in numpy arrays--##
    temperature_dfs.append(temperature_df[['Ptemp', 'Latitude', 'Temperature']])
    RH_dfs.append(RH_df[['Ptemp', 'Latitude', 'RH']])

###########################
##--Prepare for Binning--##
###########################

##--Binning for temperature data--##
all_latitudes_temperature = np.concatenate([df["Latitude"].values for df in temperature_dfs])
all_ptemps_temperature = np.concatenate([df["Ptemp"].values for df in temperature_dfs])
all_temperature = np.concatenate([df["Temperature"].values for df in temperature_dfs])
 
lat_bin_edges_temperature = np.linspace(all_latitudes_temperature.min(), all_latitudes_temperature.max(), num_bins_lat + 1)
ptemp_bin_edges_temperature = np.linspace(all_ptemps_temperature.min(), all_ptemps_temperature.max(), num_bins_ptemp + 1)
 
temperature_bin_medians, _, _, _ = binned_statistic_2d(all_latitudes_temperature, all_ptemps_temperature, 
    all_temperature, statistic="median", bins=[lat_bin_edges_temperature, ptemp_bin_edges_temperature])
 
##--Binning for RH data--##
all_latitudes_RH = np.concatenate([df["Latitude"].values for df in RH_dfs])
all_ptemps_RH = np.concatenate([df["Ptemp"].values for df in RH_dfs])
all_RH = np.concatenate([df["RH"].values for df in RH_dfs])
 
lat_bin_edges_RH = np.linspace(all_latitudes_RH.min(), all_latitudes_RH.max(), num_bins_lat + 1)
ptemp_bin_edges_RH = np.linspace(all_ptemps_RH.min(), all_ptemps_RH.max(), num_bins_ptemp + 1)
 
RH_bin_medians, _, _, _ = binned_statistic_2d(all_latitudes_RH, all_ptemps_RH, 
    all_RH, statistic="median", bins=[lat_bin_edges_RH, ptemp_bin_edges_RH])
 
################
##--PLOTTING--##
################
 
def plot_curtain(bin_medians, x_edges, y_edges, vmin, vmax, title, cbar_label):
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
 
    ##--Set axis labels and title--##
    ax.set_xlabel("Latitude (°)", fontsize=12)
    ax.set_ylabel("Altitude (m)", fontsize=12)
    ax.set_title(title)
    #ax.set_ylim(0, 6250)
    #ax.set_xlim(79.5, 83.7)
 
    ##--Save the plot--##
    #plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.tight_layout()
    plt.show()
 
##--Plot for temperature--##
plot_curtain(temperature_bin_medians, lat_bin_edges_temperature, ptemp_bin_edges_temperature, vmin=0, vmax=300,
    title=f"Temperature - PAMARCMiP {flight.replace('Flight', 'Flight ')} ({flight_date})", 
    cbar_label="Temperature (K)")
    #output_path=f"{output_path}\\CPC10/AltitudeLatitude/MultiFlights.png")

##--Plot for RH--##
plot_curtain(RH_bin_medians, lat_bin_edges_RH, ptemp_bin_edges_RH, vmin=0, vmax=110,
    title=f"Relative Humidity - PAMARCMiP {flight.replace('Flight', 'Flight ')} ({flight_date})", 
    cbar_label="RH w.r.t. Water (%)")
    #output_path=f"{output_path}\\N_10_89/AltitudeLatitude/MultiFlights.png")

########################
##--Diagnostic Plots--##
########################

##--Remove hashtags below to comment out this section--##
#'''
 
##--Counts per bin for temperature--##
temperature_bin_counts, _, _, _ = binned_statistic_2d(all_latitudes_temperature, all_ptemps_temperature, all_temperature,
    statistic="count", bins=[lat_bin_edges_temperature, ptemp_bin_edges_temperature])
 

##--Counts per bin for RH--##
RH_bin_counts, _, _, _ = binned_statistic_2d(all_latitudes_RH, all_ptemps_RH, all_RH,
    statistic="count", bins=[lat_bin_edges_RH, ptemp_bin_edges_RH])

##--Plotting--##

def plot_curtain(bin_counts, x_edges, y_edges, vmin, vmax, title, cbar_label):
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

    ##--Set axis labels and title--##
    ax.set_xlabel("Latitude (°)", fontsize=12)
    ax.set_ylabel("Altitude (m)", fontsize=12)
    ax.set_title(title)
    #ax.set_ylim(0, 6250)
    #ax.set_xlim(79.5, 83.7)
 
    ##--Save the plot--##
    #plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.tight_layout()
    plt.show()
 
##--Plot for temperature--##
plot_curtain(temperature_bin_counts, lat_bin_edges_temperature, ptemp_bin_edges_temperature, vmin=1, vmax=7500,  
    title="Particles >10 nm Data Point Counts", cbar_label="Number of Data Points")

##--Plot for RH--##
plot_curtain(RH_bin_counts, lat_bin_edges_RH, ptemp_bin_edges_RH, vmin=1, vmax=3500,  
    title="10-89 nm Data Point Counts", cbar_label="Number of Data Points")

#'''
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 10:23:01 2026

@author: repooley
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from scipy.stats import binned_statistic_2d
from datetime import date

##################
##--Open Files--##
##################

##--Set the base directory to project folder--##
directory = r"C:\Users\repooley\REP_PhD\Arctic_NPF\PAMARCMiP2012\data"

##--Select flight (Flight1 thru Flight9)--##
flights_to_analyze = ["Flight1", "Flight2", "Flight3", "Flight6", 
                      "Flight7", "Flight8", "Flight9"]

##--Set binning for Altitude and Latitude--##
num_bins_lat = 8
num_bins_ptemp = 8

################################
##--Open Files and pull data--##
################################

##--Define function that creates datasets from filenames--##
def find_files(directory, flight, partial_name):
    ##--flight data are stored in a folder called "raw"--##
    flight_dir = os.path.join(directory, "raw", flight)
    search_pattern = os.path.join(flight_dir, f"*{partial_name}*")
    return sorted(glob.glob(search_pattern))

for flight in flights_to_analyze:
    
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
    temperature_df = pd.DataFrame({'ptemp': potential_temp, 'Latitude': latitude, 
                            'Temperature': temperature}).dropna()
  
    RH_df = pd.DataFrame({'ptemp': potential_temp, 'Latitude': latitude, 
                           'RH': RH}).dropna()

    ###########################
    ##--Create 2D histogram--##
    ###########################
    
    ##--Compute global min/max values across all data BEFORE dropping NaNs--##
    lat_min, lat_max = np.nanmin(latitude), np.nanmax(latitude)
    ptemp_min, ptemp_max = np.nanmin(potential_temp), np.nanmax(potential_temp)
    
    ##--Generate common bin edges using specified number of bins--##
    common_lat_bin_edges = np.linspace(lat_min, lat_max, num_bins_lat + 1)
    common_ptemp_bin_edges = np.linspace(ptemp_min, ptemp_max, num_bins_ptemp + 1)
    
    ##--Make 2D histograms using common bins--##
    ##--Temperature--##
    temperature_bin_medians, _, _, _ = binned_statistic_2d(temperature_df['Latitude'], 
        temperature_df['ptemp'], temperature_df['Temperature'], statistic='median', 
        bins=[common_lat_bin_edges, common_ptemp_bin_edges])
    
    ##--RH--##
    RH_bin_medians, _, _, _ = binned_statistic_2d(RH_df['Latitude'],
        RH_df['ptemp'], RH_df['RH'], statistic='median', 
        bins=[common_lat_bin_edges, common_ptemp_bin_edges])

    ################
    ##--PLOTTING--##
    ################

    ##--Make special color map where 0 values are white--##
    new_cmap = plt.get_cmap('viridis')
    ##--Values under specified minimum will be white--##
    new_cmap.set_under('w')
    
    ##--Particles larger than 10 nm--##
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    
    ##--Use pcolormesh for the plot, set minimum for viridis colors as 1--##
    temperature_plot = ax1.pcolormesh(common_lat_bin_edges, common_ptemp_bin_edges, temperature_bin_medians.T,  # Transpose to align correctly
        shading='auto', cmap=new_cmap, vmin=0, vmax=300)
    
    ##--Add colorbar--##
    cb1 = fig1.colorbar(temperature_plot, ax=ax1)
    cb1.minorticks_on()
    cb1.set_label('Temperature (K) $', fontsize=12)
    
    ##--Set axis labels--##
    ax1.set_xlabel('Latitude (°)', fontsize=12)
    ax1.set_ylabel('Potential Temperature (K)', fontsize=12)
    ax1.set_title(f"Temperature - PAMARCMiP {flight.replace('Flight', 'Flight ')} ({flight_date})")
    #ax1.set_ylim(0, 6250)
    #ax1.set_xlim(79.5, 83.7)
    
    ##--Use f-string to save file with flight# appended--##
    #CPC10_output_path = f"{output_path}\\/{flight}"
    #plt.savefig(CPC10_output_path, dpi=600, bbox_inches='tight') 
    
    plt.tight_layout()
    plt.show()
    
    
    ##--10-60 nm: Aitken mode--##
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    
    ##--Use pcolormesh for the plot, set minimum for viridis colors as 1--##
    RH_plot = ax2.pcolormesh(common_lat_bin_edges, common_ptemp_bin_edges, RH_bin_medians.T,  # Transpose to align correctly
        shading='auto', cmap=new_cmap, vmin=0, vmax=110)
    
    ##--Add colorbar--##
    cb2 = fig2.colorbar(RH_plot, ax=ax2)
    cb2.minorticks_on()
    cb2.set_label('RH w.r.t. Water (%)', fontsize=12)
    
    ##--Set axis labels--##
    ax2.set_xlabel('Latitude (°)', fontsize=12)
    ax2.set_ylabel('Potential Temperature (K)', fontsize=12)
    ax2.set_title(f"Relative Humidity - PAMARACMiP {flight.replace('Flight', 'Flight ')} ({flight_date})")
    #ax2.set_ylim(0, 6250)
    #ax2.set_xlim(79.5, 83.7)
    
    ##--Use f-string to save file with flight# appended--##
    #CPC10_output_path = f"{output_path}\\/{flight}"
    #plt.savefig(CPC10_output_path, dpi=600, bbox_inches='tight') 
    
    plt.tight_layout()
    plt.show()
 
    ########################
    ##--Diagnostic Plots--##
    ########################
    
    ##--Remove hashtags below to comment out this section--##
    
    ##--Counts per bin for CPC10 data--##
    temperature_bin_counts, _, _, _ = binned_statistic_2d(temperature_df['Latitude'], 
        temperature_df['ptemp'], temperature_df['Temperature'], statistic='count', 
        bins=[common_lat_bin_edges, common_ptemp_bin_edges])
     
    ##--Counts per bin for N3-10 particles--##
    RH_bin_counts, _, _, _ = binned_statistic_2d(RH_df['Latitude'], 
        RH_df['ptemp'], RH_df['RH'], statistic='count', 
        bins=[common_lat_bin_edges, common_ptemp_bin_edges])
    
    
    ##--Plotting--##
    
    ##--Make special color map where 0 values are white--##
    new_cmap = plt.get_cmap('inferno')
    ##--Values under specified minimum will be white--##
    new_cmap.set_under('w')
    
    ##--Particles larger than 10 nm--##
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    
    ##--Use pcolormesh for the plot, set minimum for viridis colors as 1--##
    temperature_diag_plot = ax2.pcolormesh(common_lat_bin_edges, common_ptemp_bin_edges, temperature_bin_counts.T,  # Transpose to align correctly
        shading='auto', cmap=new_cmap, vmin=1, vmax=2000)
    
    ##--Add colorbar--##
    cb2 = fig2.colorbar(temperature_diag_plot, ax=ax2)
    cb2.minorticks_on()
    cb2.set_label('Number of Data Points', fontsize=12)
    
    ##--Set axis labels--##
    ax2.set_xlabel('Latitude (°)', fontsize=12)
    ax2.set_ylabel('Potential Temperature (K)', fontsize=12)
    ax2.set_title(f"Temperature - PAMARCMiP {flight.replace('Flight', 'Flight ')} ({flight_date})")
    #ax2.set_ylim(0, 6250)
    #ax2.set_xlim(79.5, 83.7)
    
    ##--Use f-string to save file with flight# appended--##
    #CPC10_diag_output_path = f"{output_path}\\CPC10/AltitudeLatitude/{flight}_diagnostic"
    #plt.savefig(CPC10_diag_output_path, dpi=600, bbox_inches='tight') 
    
    plt.tight_layout()
    plt.show()

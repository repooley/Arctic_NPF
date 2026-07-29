# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 09:58:08 2026

@author: repooley
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from datetime import date
import cartopy.crs as ccrs
import cartopy.feature as cfeature

###################
##--User inputs--##
###################

##--Set the base directory to project folder--##
directory = r"C:\Users\repooley\REP_PhD\Arctic_NPF\FIREACE1998\data"

##--Flights to analyze - flights 1-18--##
flights_to_analyze = ["Flight3",  
                      "Flight7", "Flight8", "Flight9", "Flight10", "Flight11", "Flight12",
                      "Flight13", "Flight14", "Flight15", "Flight16", "Flight17", "Flight18"]

##--Base output path in directory--##
output_path = r"C:\Users\repooley\REP_PhD\Arctic_NPF\FIRACE1998\data\processed\VerticallyBinnedData"

PCASP_bins_path = r"C:\Users\repooley\REP_PhD\Arctic_NPF\FIREACE1998\data\raw\FIREACE1998_PCASP_bins.csv"

#%%

################################
##--Open Files and Pull Data--##
################################

##--Define function that creates datasets from filenames--##
def find_files(directory, flight, partial_name):
    ##--flight data are stored in a folder called "raw"--##
    flight_dir = os.path.join(directory, "raw", flight)
    search_pattern = os.path.join(flight_dir, f"*{partial_name}*")
    return sorted(glob.glob(search_pattern))

for flight in flights_to_analyze:

    ##--'raw' contains a 1hz and 2min datafile, the 1hz one is always first--##
    data = pd.read_csv(find_files(directory, flight, "FIREACE")[0])
    
    ##--Pull data variables from file--##
    time = data['Time'] # HHMMSS UTC time
    pressure = data['Pressure'] * 100 # in Pa
    temperature = data['Temperature'] + 273.15 # in K
    RH = data['RH'] # percent wrt water
    altitude = data['Altitude'] # in m (agl?)
    latitude = data['Latitude'] # degrees
    longitude = data['Longitude'] # degrees
    
    ##--There are some erraneously negative latitudes--##
    latitude = np.where(latitude >= 0, latitude, np.nan)

    ##--Particle data, 3 and 10 nm cutoffs, respectively--##
    CPC3_data = data['CN3025_corrected'] # Uncorrected data has a flow issue
    CPC10_data = data['CN7610']
    
    averaged_data = pd.read_csv(find_files(directory, flight, "FIREACE")[1])
    averaged_data = averaged_data.set_index('Time', drop=False)

    PCASP_bins = pd.read_csv(PCASP_bins_path)

    PCASP_data = averaged_data.iloc[:, 14:29] # select PCASP data

    ##--Add time, total_num to UHSAS_bins df--##
    PCASP_data.insert(0, 'Time', averaged_data['Time'])

    ##--Set time as the index for later alignment--##
    PCASP_data = PCASP_data.set_index('Time')

    ##--15 total bins--##
    PCASP_bin_num = [f'bin_{i}' for i in range(1, 16)]

    ##--Information for bins--##
    PCASP_bin_center = PCASP_bins['bin_avg']
    PCASP_lower_bound = PCASP_bins['lower_bound']
    PCASP_upper_bound = PCASP_bins['upper_bound']

    ##--Put column names and content in a dictionary and then convert to a Pandas df--##
    PCASP_df = pd.DataFrame({col: PCASP_data[col] for col in PCASP_bin_num})

    ##--Create new column names by rounding the bin center values to the nearest integer--##
    PCASP_new_col_names = PCASP_bin_center.round().astype(int).tolist()

    ##--Rename the PCASP_bins df columns to bin average values--##
    PCASP_data.columns = PCASP_new_col_names

    ##--Nans are denoted by -8888--##
    
    ####################################
    ##--Assign date to flight number--##
    ####################################
    
    if flight=="Flight1":
        flight_date = date(1998, 4, 8)
    elif flight=="Flight2":
        flight_date = date(1998, 4, 9)
    elif flight=="Flight3":
        flight_date = date(1998, 4, 12)
    elif flight=="Flight4":
        flight_date = date(1998, 4, 14)
    elif flight=="Flight5":
        flight_date = date(1998, 4, 15)
    elif flight=="Flight6":
        flight_date = date(1998, 4, 16)
    elif flight=="Flight7" or flight=="Flight8": 
        flight_date = date(1998, 4, 17)
    elif flight=="Flight9": 
        flight_date = date(1998, 4, 18)
    elif flight=="Flight10" or flight=="Flight11": 
        flight_date = date(1998, 4, 21)
    elif flight=="Flight12":
        flight_date = date(1998, 4, 22)
    elif flight=="Flight13":
        flight_date = date(1998, 4, 24)
    elif flight=="Flight14":
        flight_date = date(1998, 4, 25)
    elif flight=="Flight15":
        flight_date = date(1998, 4, 27)
    elif flight=="Flight16" or flight=="Flight17": 
        flight_date = date(1998, 4, 28)
    elif flight=="Flight18": 
        flight_date = date(1998, 4, 29)
    
    #%%
    ######################
    ##--Calc N(2.5-10)--##
    ######################
    
    ##--Convert to STP!--##
    P_STP = 101325  # Pa
    T_STP = 273.15  # K
    
    ##--Create empty list for CPC3 particles--##
    CPC3_conc_STP = []
    
    for CPC3, T, P in zip(CPC3_data, temperature, pressure):
        if np.isnan(CPC3) or np.isnan(T) or np.isnan(P):
            ##--Append with NaN if any input is NaN--##
            CPC3_conc_STP.append(np.nan)
        else:
            ##--Perform conversion if all inputs are valid--##
            CPC3_conversion = CPC3 * (P_STP / P) * (T / T_STP)
            CPC3_conc_STP.append(CPC3_conversion)
        
    ##--Create empty list for CPC10 particles--##
    CPC10_conc_STP = []
    
    for CPC10, T, P in zip(CPC10_data, temperature, pressure):
        if np.isnan(CPC10) or np.isnan(T) or np.isnan(P):
            ##--Append with NaN if any input is NaN--##
            CPC10_conc_STP.append(np.nan)
        else:
            ##--Perform conversion if all inputs are valid--##
            CPC10_conversion = CPC10 * (P_STP / P) * (T / T_STP)
            CPC10_conc_STP.append(CPC10_conversion)
    
    ##--Creates a Pandas dataframe for particle data--##
    df = pd.DataFrame({'Altitude': altitude, 'CPC3_conc':CPC3_conc_STP, 'CPC10_conc': CPC10_conc_STP})
    
    ##--Calculate N3-10 particles--##
    nuc_particles = (df['CPC3_conc'] - df['CPC10_conc'])
    
    ##--Change calculated particle counts less than zero to NaN--##
    nuc_particles = np.where(nuc_particles >= 0, nuc_particles, np.nan)
    
    ################
    ##--Plotting--##
    ################
    
    ##--Create the map with a North Polar Stereo projection and orient map so North America is at bottom--##
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=-90)})
    
    ##--Add land and ocean features--##
    ##--Physical specifies natural elements. 10m resolution is very fine--##
    land = cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='black', 
                                        facecolor='lightgray')
    ocean = cfeature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='none', 
                                         facecolor='lightblue')
    ##--Add features to plot ax--##
    ax.add_feature(ocean)  # Ocean in light blue
    ax.add_feature(land)   # Land in light gray
    '''
    if flight == 'Flight1': 
        ax.set_extent([0, 25, 76, 82], crs=ccrs.PlateCarree())
    elif flight == 'Flight8' or flight == 'Flight9' or flight == 'Flight10': 
        ax.set_extent([-145, -120, 65, 72], crs=ccrs.PlateCarree())
    else:
        ax.set_extent([-100, -68, 79, 84], crs=ccrs.PlateCarree())
    '''
    ##--Plot flight track colored by altitude. Change Z-order so appears on top of map--##
    track = ax.scatter(longitude, latitude, c=nuc_particles, cmap='viridis', transform=ccrs.PlateCarree(), s=10, zorder=3)
    
    ##--Add gridlines--##
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = True
    gl.bottom_labels = True
    
    ##--Use f-string to embed flight # variable in plot title--##
    plt.title(f"2.5-10 nm Particles Along Track - FIRE-ACE {flight.replace('Flight', 'Flight ')} ({flight_date})", fontsize=16)
    
    ##--Add colorbar--##
    cbar = plt.colorbar(track, ax=ax, orientation='vertical', shrink=0.7)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label(label="2.5-10 nm Particles $(Counts/cm^{3})$", fontsize=14)
    
    plt.show()
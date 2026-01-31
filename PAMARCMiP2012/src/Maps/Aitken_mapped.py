# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 10:37:11 2026

@author: repooley
"""

import os
import glob
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from datetime import date

#########################
##--Open ICARTT Files--##
#########################

##--Set the base directory to project folder--##
directory = r"C:\Users\repooley\REP_PhD\Arctic_NPF\PAMARCMiP2012\data"
 
##--Define function that creates datasets from filenames--##
def find_files(directory, flight, partial_name):
    ##--flight data are stored in a folder called "raw"--##
    flight_dir = os.path.join(directory, "raw", flight)
    search_pattern = os.path.join(flight_dir, f"*{partial_name}*")
    return sorted(glob.glob(search_pattern))
 
##--Choose which flights to analyze here!--##

PAMARCMiP_to_analyze = [
    "Flight1", 
    "Flight2", 
    "Flight3",
    "Flight4",
    "Flight5", 
    "Flight6", 
    "Flight7",
    "Flight8",
    "Flight9"
    ]

PAMARCMiP_dfs = []

for flight in PAMARCMiP_to_analyze:  
    
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
    longitude = data['Longitude'] # in degrees
    temperature = data['Temp'] + 273.15 # in K
    pressure = data['Pressure'] # in pa
    time = data['Time'] # seconds since midnight
    RH = data['RH']
    
    ##--The first datapoint in 'latitude' column is erraneous (47.12 N)--##
    latitude = latitude.where(latitude >= 50, np.nan)
    
    ##--USHAS Data--##
    UHSAS_total_num = data['UH-TotConc'] # particles/cm^3
    
    ##--Bin data are in a CSV file--##
    ##--SAME bins for NETCARE and PAMARCMiP--##
    UHSAS_bins = pd.read_csv(r"C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data\raw\NETCARE2015_UHSAS_bins.csv")
    
    ##--Make list of columns to pull, each named bin_x--##
    UHSAS_bin_num = [f'UH_bin_{i}' for i in range(1, 100)]
    
    ##--Information for bins 1 thru 99--##
    UHSAS_bin_center = UHSAS_bins['bin_avg'].iloc[0:100]
    UHSAS_lower_bound = UHSAS_bins['lower_bound'].iloc[0:100]
    UHSAS_upper_bound = UHSAS_bins['upper_bound'].iloc[0:100]
    
    ##--Put column names and content in a dictionary and then convert to a Pandas df--##
    UHSAS_bins = pd.DataFrame({col: data[col] for col in UHSAS_bin_num})
    
    ##--Create new column names by rounding the bin center values to the nearest integer--##
    UHSAS_new_col_names = UHSAS_bin_center.round().astype(int).tolist()
    
    ##--Rename the UHSAS_bins df columns to bin average values--##
    UHSAS_bins.columns = UHSAS_new_col_names
    
    ##--10 nm CPC data--##
    CPC10_conc = data['CPC10'] # count/cm^3
    
    ##--REMOVE OUTLIERS above 99th percentile--##
    p = 0.99
    
    ##--Compute threshold for each UHSAS column--##
    uhsas_thresh = UHSAS_bins.quantile(p)
    
    ##--keep only rows where each bin is below its threshold--##
    UHSAS_bins_filtered = UHSAS_bins[UHSAS_bins.le(uhsas_thresh, axis=1)]
    
    cpc10_thresh = CPC10_conc.quantile(p)
    CPC10_filtered = CPC10_conc[CPC10_conc <= cpc10_thresh]
    
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
    
    ###############################
    ##--De-Normalize UHSAS Data--##
    ###############################
    
    ##--For total count calculation--##
    
    ##--Calculate dlogDp for UHSAS bins--##
    UHSAS_dlogDp = np.log(UHSAS_upper_bound.values) - np.log(UHSAS_lower_bound.values)
    
    ##--Get only particle count data (excluding 'Time')--##
    UHSAS_particle_counts = UHSAS_bins.loc[:, UHSAS_new_col_names]  # Adjust column names as needed
    
    ##--De-Normalize counts by multiplying by dlogDp across all rows--##
    UHSAS_denorm_counts = UHSAS_particle_counts.multiply(UHSAS_dlogDp, axis=1)
    
    ##################
    ##--UHSAS bins--##
    ##################
 
    ##--Bin data are in a CSV file--##
    ##--SAME bins for NETCARE and PAMARCMiP--##
    UHSAS_bins = pd.read_csv(r"C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data\raw\NETCARE2015_UHSAS_bins.csv")
    
    ##--Make list of columns to pull, each named bin_x--##
    UHSAS_bin_num = [f'UH_bin_{i}' for i in range(1, 100)]
    
    ##--Information for bins 1 thru 99--##
    UHSAS_bin_center = UHSAS_bins['bin_avg'].iloc[0:100]
    UHSAS_lower_bound = UHSAS_bins['lower_bound'].iloc[0:100]
    UHSAS_upper_bound = UHSAS_bins['upper_bound'].iloc[0:100]
    
    ##--Put column names and content in a dictionary and then convert to a Pandas df--##
    UHSAS_bins = pd.DataFrame({col: data[col] for col in UHSAS_bin_num})
    
    ##--Create new column names by rounding the bin center values to the nearest integer--##
    UHSAS_new_col_names = UHSAS_bin_center.round().astype(int).tolist()
    
    ##--Rename the UHSAS_bins df columns to bin average values--##
    UHSAS_bins.columns = UHSAS_new_col_names
    
    ##--REMOVE OUTLIERS above 99th percentile--##
    p = 0.99
    
    ##--Compute threshold for each UHSAS column--##
    uhsas_thresh = UHSAS_bins.quantile(p)
    
    ##--keep only rows where each bin is below its threshold--##
    UHSAS_bins_filtered = UHSAS_bins[UHSAS_bins.le(uhsas_thresh, axis=1)]
 
    #####################
    ##--Calc N(10-60)--##
    #####################
    
    ##--Create df with UHSAS total counts--##
    UHSAS_total = pd.DataFrame({'Time': time, 'Total_count': UHSAS_total_num})
    
    ##--Create df with CPC10 counts and set index to time--##
    CPC10_counts = pd.DataFrame({'Time':time, 'Counts':CPC10_filtered})
    
    ##--Calculate particles below UHSAS lower cutoff--##
    n_10_60 = (CPC10_counts['Counts'] - UHSAS_total['Total_count'])
    
    ##--Change calculated particle counts less than zero to NaN--##
    n_10_60 = np.where(n_10_60 >= 0, n_10_60, np.nan)
    
    ##--Put N(10-60) bin center in a df--##
    n_10_60_center = pd.DataFrame([35])
    
    ##--Flatten--##
    n_10_60_center = pd.Series(n_10_60_center.values.flatten())
    
    ##--Convert n_10_60 to a df--##
    n_10_60_df = pd.DataFrame({'35': n_10_60, 'time':time}).set_index('time')
    
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

    ##--Plot flight track colored by altitude. Change Z-order so appears on top of map--##
    track = ax.scatter(longitude, latitude, c=n_10_60_df['35'], cmap='viridis', transform=ccrs.PlateCarree(), s=10, zorder=3)
    
    ##--Add gridlines--##
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = True
    gl.bottom_labels = True
    
    ##--Use f-string to embed flight # variable in plot title--##
    plt.title(f"10-60 nm Particles Along Track - FIRE-ACE {flight.replace('Flight', 'Flight ')} ({flight_date})", fontsize=16)
    
    ##--Add colorbar--##
    cbar = plt.colorbar(track, ax=ax, orientation='vertical', shrink=0.7)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label(label="10-60 nm Particles $(Counts/cm^{3})$", fontsize=14)
    
    plt.show()
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 13:09:19 2025

@author: repooley
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.ticker as mticker

##################
##--Open Files--##
##################

##--Set the base directory to project folder--##
directory = r"C:\Users\repooley\REP_PhD\Arctic_NPF\PAMARCMiP2012\data\raw"

##--Select flight (Flight1 thru Flight9)--##
flights_to_analyze = ["Flight1", "Flight2", "Flight3", "Flight4", "Flight5", "Flight6", 
                      "Flight7", "Flight8", "Flight9"]

##--Define function that creates datasets from filenames--##
def find_files(directory, flight, partial_name):
    flight_dir = os.path.join(directory, flight)
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
    
    ##--REMOVE OUTLIERS above 99.5th percentile--##
    p = 0.995
    
    ##--Compute threshold for each UHSAS column--##
    uhsas_thresh = UHSAS_bins.quantile(p)
    
    ##--keep only rows where each bin is below its threshold--##
    UHSAS_bins_filtered = UHSAS_bins[UHSAS_bins.le(uhsas_thresh, axis=1)]
    
    cpc10_thresh = CPC10_conc.quantile(p)
    CPC10_filtered = CPC10_conc[CPC10_conc <= cpc10_thresh]
    
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
        
    PTemp_series = pd.Series(potential_temp, index=time)
    
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
    n_10_60 = pd.DataFrame({'49.5': n_10_60, 'time':time}).set_index('time')
    
    ################
    ##--Plotting--##
    ################
    
    ##--Concatenate bin centers and reindex--##
    bin_centers = pd.concat([n_10_60_center, UHSAS_bin_center], axis=0).reset_index(drop=True)
    
    ##--Concatenate bin edges--##
    combined_bin_edges_optical = np.concatenate([
        UHSAS_upper_bound.values,  # UHSAS bins continue from 85
    ])
    
    
    ##--Create df containing UHSAS columns--##
    optical_bins_aligned = pd.concat([UHSAS_bins_filtered], axis=1)
    
    ##--Apply rolling average to particle data--##
    optical_bins_smoothed = optical_bins_aligned.rolling(window=5, min_periods=1, center=True).mean()
    
    ##--Compute the median, 75th, and 90th percentiles of data--##
    ##--Apply smoothing--##
    optical_bins_median = optical_bins_aligned.median(axis=0).rolling(window=5, min_periods=1, center=True).mean()
    optical_bins_75th = optical_bins_aligned.quantile(q=0.75, axis=0).rolling(window=5, min_periods=1, center=True).mean()
    optical_bins_25th = optical_bins_aligned.quantile(q=0.25, axis=0).rolling(window=5, min_periods=1, center=True).mean()
    optical_bins_max = optical_bins_aligned.max(axis=0).rolling(window=5, min_periods=1, center=True).mean()
    optical_bins_min = optical_bins_aligned.min(axis=0).rolling(window=5, min_periods=1, center=True).mean()
    
    
    n_10_60_median = n_10_60.median()
    n_10_60_75th = n_10_60.quantile(q=0.75)
    n_10_60_25th = n_10_60.quantile(q=0.25)
    n_10_60_max = n_10_60.max()
    n_10_60_min = n_10_60.min()
    
    
    ##--Set up figure--##
    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    
    ##--Add percentile ranges--##
    ax.fill_between(combined_bin_edges_optical, optical_bins_min, optical_bins_max, color='orchid', alpha=0.4)
    ax.fill_between(combined_bin_edges_optical, optical_bins_25th, optical_bins_75th, color='orchid', alpha=1)
    
    ax.fill_between(n_10_60_center, n_10_60_min, n_10_60_max, color='orchid', alpha=0.4, linewidth=3, label='Full Range')
    ax.fill_between(n_10_60_center, n_10_60_25th, n_10_60_75th, color='orchid', alpha=1, linewidth=3, label='Interquartile Range')
    
    ax.vlines(x=10, ymin=-250, ymax=4000, colors='darkgrey', linewidth=1.5, linestyle='--')
    ax.vlines(x=60, ymin=-250, ymax=4000, colors='darkgrey', linewidth=1.5, linestyle='--')
    
    ##--Add medians--##
    ax.plot(combined_bin_edges_optical, optical_bins_median, c='indigo', linewidth=2, label='Median')
    ax.plot(n_10_60_center, n_10_60_median, marker='.', c='indigo', markersize=8)
    
    plt.xticks([10, 60, 100, 1000], fontsize=16)
    plt.yticks(fontsize=16)
    
    ##--Format x-axis on a log scale--##
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    
    ##--Format y-axis to leave extra space at the bottom--##
    plt.ylim(-100, 1500)
    
    plt.xlabel('Dp (nm)', fontsize=16)
    plt.ylabel('dN/dlogDp', fontsize=16)
    plt.title("PAMARCMiP Particle Number Size Distribution", fontsize=16)
    plt.legend(fontsize=14)
    
    plt.show()
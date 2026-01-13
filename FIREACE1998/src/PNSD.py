# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 12:33:25 2025

@author: repooley
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic_2d
import matplotlib.pyplot as plt 
import matplotlib.ticker as mticker
from datetime import date

###################
##--User inputs--##
###################

##--Set the base directory to project folder--##
directory = r"C:\Users\repooley\REP_PhD\Arctic_NPF\FIREACE1998\data\raw"

##--Flights to analyze - flights 1-18--##
flights_to_analyze = ["Flight3",  
                      "Flight7", "Flight8", "Flight9", "Flight10", "Flight11", "Flight12",
                      "Flight13", "Flight14", "Flight15", "Flight16", "Flight17", "Flight18"]

##--Set binning for PTemp and Latitude--##
##--Define number of bins here--##
num_bins_lat = 6
num_bins_ptemp = 12

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

#################
##--Pull data--##
#################

for flight in flights_to_analyze:
    
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
    

    ##--Pull csv file containing all data--##
    files = find_files(directory, flight, "FIREACE")
    
    ##--The averaged data is always the second file--##
    if files:
        data = pd.read_csv(files[1])
    
    ##--Pull data variables from file--##
    time = data['Time'] # HHMMSS UTC time
    pressure = data['Pressure'] * 100 # in Pa
    temperature = data['Temperature'] + 273.15 # in K
    RH = data['RH'] # percent wrt water
    altitude = data['Altitude'] # in m (agl?)
    latitude = data['Latitude'] # degrees
    #longitude = data['Longitude'] # degrees
    
    ##--Particle data, 3 and 10 nm cutoffs, respectively--##
    CPC3_data = data['CN3025'] # Uncorrected data has a flow issue - but corrected not populated for many flights
    CPC10_data = data['CN7610']
    
    PCASP_bins = pd.read_csv(PCASP_bins_path)
    
    PCASP_data = data.iloc[:, 14:29] # select PCASP data
    
    ##--Add time, total_num to UHSAS_bins df--##
    PCASP_data.insert(0, 'Time', data['Time'])
    
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
    particle_df = pd.DataFrame({'Altitude': altitude, 'Latitude': latitude,
                       'CPC3_conc':CPC3_conc_STP, 'CPC10_conc': CPC10_conc_STP})
    
    ##--Calculate N3-10 particles--##
    nuc_particles = (particle_df['CPC3_conc'] - particle_df['CPC10_conc'])
    
    ##--Change calculated particle counts less than zero to NaN--##
    nuc_particles = np.where(nuc_particles >= 0, nuc_particles, np.nan)
    
    ##--Add nucleating particles to df--##
    particle_df['n_3_10'] = nuc_particles
    
    ##--Put N(3-10) bin center in a df--##
    n_3_10_center = pd.DataFrame([6.5])
    
    ############################
    ##--Normalize PCASP data--##
    ############################
    
    ##--Calculate dlogDp for each bin in numpy array--##
    dlogDp = np.log(PCASP_upper_bound.values) - np.log(PCASP_lower_bound.values)
    
    ##--Get only particle count data (excluding 'Time')--##
    PCASP_particle_counts = PCASP_data.loc[:, PCASP_new_col_names]
    
    ##--Normalize counts by dividing by dlogDp across all rows--##
    PCASP_dNdlogDp = PCASP_data.divide(dlogDp, axis=1)
    
    ##--Convert to STP!--##
    P_STP = 101325  # Pa
    T_STP = 273.15  # K
    
    ##--Create empty list for PCASP particles--##
    PCASP_STP = []
    
    for PCASP, T, P in zip(PCASP_dNdlogDp.values, data['Temperature']+273.15, data['Pressure']*100):
        if np.isnan(T) or np.isnan(P):
            ##--Append with NaN if any input is NaN--##
            PCASP_STP.append([np.nan]*len(PCASP))
        else:
            ##--Perform conversion if all inputs are valid--##
            corrected_PCASP = PCASP * (P_STP / P) * (T / T_STP)
            PCASP_STP.append(corrected_PCASP)
    
    ##--Convert back to DataFrame with same columns and index--##
    PCASP_STP = pd.DataFrame(PCASP_STP, columns=PCASP_dNdlogDp.columns, index=particle_df.index)
    
    ##--Add PCASP data to the dataframe--##
    particle_df = pd.concat([particle_df, PCASP_STP], axis=1)
    
    ##--Add PCASP total counts to the dataframe--##
    particle_df['PCTcon'] = data['PCTcon']
    
    ######################
    ##--Calc N(10-150)--##
    ######################
    
    ##--Calculate particles below UHSAS lower cutoff--##
    n_10_150 = (particle_df['CPC10_conc'] - particle_df['PCTcon'])
    
    ##--Change calculated particle counts less than zero to NaN--##
    n_10_150 = np.where(n_10_150 >= 0, n_10_150, np.nan)
    
    ##--Put N(10-130) bin center in a df--##
    n_10_150_center = pd.DataFrame([70])
    
    particle_df['n_10_130'] = n_10_150
    
    ##--Compute TOTAL counts from all size bins combined--##
    particle_df['Total_particles_STP'] = (particle_df['n_3_10'].fillna(0) + 
          particle_df['n_10_130'].fillna(0) + particle_df['PCTcon'].fillna(0))
    
    ###################################
    ##--Normalize nuc and grow bins--##
    ###################################
    
    ##--Calculate dlogDp--##
    dlog_3_10 = np.log(10.0) - np.log(2.5)    # for 2.5 - 10 nm
    dlog_10_150 = np.log(150.0) - np.log(10.0)  # for 10 - 130 nm 
    
    ##--Create dN/dlogDp--##
    n_3_10_dNdlogDp = nuc_particles / dlog_3_10
    n_10_150_dNdlogDp = n_10_150 / dlog_10_150
    
    ##--Put into dataframes--##
    n_3_10_dNdlogDp = pd.DataFrame(n_3_10_dNdlogDp)
    n_10_150_dNdlogDp = pd.DataFrame(n_10_150_dNdlogDp)
    
    ################
    ##--Plotting--##
    ################
    
    ##--Concatenate bin centers and reindex--##
    bin_centers = pd.concat([n_3_10_center, n_10_150_center, PCASP_bin_center], axis=0).reset_index(drop=True)
    
    ##--Concatenate bin edges--##
    combined_bin_edges_optical = np.concatenate([PCASP_upper_bound.values])
    
    ##--Create df containing UHSAS columns--##
    optical_bins_aligned = pd.concat([PCASP_STP], axis=1)
    
    ##--Apply rolling average to particle data--##
    optical_bins_smoothed = optical_bins_aligned.rolling(window=5, min_periods=1, center=True).mean()
    
    ##--Compute the median, 75th, and 90th percentiles of data--##
    ##--Apply smoothing--##
    optical_bins_median = optical_bins_aligned.median(axis=0).rolling(window=5, min_periods=1, center=True).mean()
    optical_bins_75th = optical_bins_aligned.quantile(q=0.75, axis=0).rolling(window=5, min_periods=1, center=True).mean()
    optical_bins_25th = optical_bins_aligned.quantile(q=0.25, axis=0).rolling(window=5, min_periods=1, center=True).mean()
    optical_bins_max = optical_bins_aligned.max(axis=0).rolling(window=5, min_periods=1, center=True).mean()
    optical_bins_min = optical_bins_aligned.min(axis=0).rolling(window=5, min_periods=1, center=True).mean()
    
    n_3_10_median = n_3_10_dNdlogDp.median()
    n_3_10_75th = n_3_10_dNdlogDp.quantile(q=0.75)
    n_3_10_25th = n_3_10_dNdlogDp.quantile(q=0.25)
    n_3_10_max = n_3_10_dNdlogDp.max()
    n_3_10_min = n_3_10_dNdlogDp.min()
    
    n_10_150_median = n_10_150_dNdlogDp.median()
    n_10_150_75th = n_10_150_dNdlogDp.quantile(q=0.75)
    n_10_150_25th = n_10_150_dNdlogDp.quantile(q=0.25)
    n_10_150_max = n_10_150_dNdlogDp.max()
    n_10_150_min = n_10_150_dNdlogDp.min()
    
    ##--Set up figure--##
    fig, ax = plt.subplots(1, 1, figsize=(12,6))
    
    ##--Add percentile ranges--##
    ax.fill_between(combined_bin_edges_optical, optical_bins_min, optical_bins_max, 
                    color='yellowgreen', alpha=0.4, edgecolor='none')
    ax.fill_between(combined_bin_edges_optical, optical_bins_25th, optical_bins_75th, 
                    color='yellowgreen', alpha=1, edgecolor='none')
    
    ##--Flatten N(3-10) to one dimension--##
    n_3_10_center = n_3_10_center.values.flatten()
    n_10_150_center = n_10_150_center.values.flatten()
    
    
    ax.fill_between(n_3_10_center, n_3_10_min, n_3_10_max, color='yellowgreen', 
                    alpha=0.4, linewidth=3, label='Full Range', edgecolor='none')
    ax.fill_between(n_3_10_center, n_3_10_25th, n_3_10_75th, color='yellowgreen', 
                    alpha=1, linewidth=3, label='Interquartile Range', edgecolor='none')
    
    ##--Fill between on x-axis to give appearance of a full bin--##
    ##--Define bin edges--##
    bin_edges = np.array([2.5, 10.0])
    bin_edges2 = np.array([10, 150])
    
    ##--Repeat y-values across the bin--##
    nuc_min_fill = np.array([n_3_10_min[0], n_3_10_min[0]])
    nuc_max_fill = np.array([n_3_10_max[0], n_3_10_max[0]])
    nuc_25_fill = np.array([n_3_10_25th[0], n_3_10_25th[0]])
    nuc_75_fill = np.array([n_3_10_75th[0], n_3_10_75th[0]])
    
    grow_min_fill = np.array([n_10_150_min[0], n_10_150_min[0]])
    grow_max_fill = np.array([n_10_150_max[0], n_10_150_max[0]])
    grow_25_fill = np.array([n_10_150_25th[0], n_10_150_25th[0]])
    grow_75_fill = np.array([n_10_150_75th[0], n_10_150_75th[0]])
    
    ##--Fill full range--##
    ax.fill_between(bin_edges, nuc_min_fill, nuc_max_fill, color='yellowgreen', alpha=0.4, linewidth=3, edgecolor='none')
    ax.fill_between(bin_edges2, grow_min_fill, grow_max_fill, color='yellowgreen', alpha=0.4, linewidth=3, edgecolor='none')
    
    ##--Fill interquartile range--##
    ax.fill_between(bin_edges, nuc_25_fill, nuc_75_fill, color='yellowgreen', alpha=1, linewidth=3, edgecolor='none')
    ax.fill_between(bin_edges2, grow_25_fill, grow_75_fill, color='yellowgreen', alpha=1, linewidth=3, edgecolor='none')
    
    ##--Add medians--##
    ax.plot(combined_bin_edges_optical, optical_bins_median, c='darkolivegreen', linewidth=2)
    
    ##--Fill n_3_10_median to edges--##
    median_fill = np.array([n_3_10_median, n_3_10_median])
    ax.plot(bin_edges, median_fill, c='darkolivegreen', linewidth=2)
    
    ##--Same for n_10_89--##
    median_fill2 = np.array([n_10_150_median, n_10_150_median])
    ax.plot(bin_edges2, median_fill2, c='darkolivegreen', linewidth=2)
    
    
    ax.fill_between(n_10_150_center, n_10_150_min, n_10_150_max, color='yellowgreen',
                    alpha=0.4, linewidth=3, edgecolor='none')
    ax.fill_between(n_10_150_center, n_10_150_25th, n_10_150_75th, color='yellowgreen', 
                    alpha=1, linewidth=3, edgecolor='none')
    
    ax.vlines(x=2.5, ymin=-250, ymax=4000, colors='darkgrey', linewidth=1.5, linestyle='--')
    ax.vlines(x=10, ymin=-250, ymax=4000, colors='darkgrey', linewidth=1.5, linestyle='--')
    ax.vlines(x=150, ymin=-250, ymax=4000, colors='darkgrey', linewidth=1.5, linestyle='--')
    
    ##--Add medians--##
    ax.plot(combined_bin_edges_optical, optical_bins_median, c='darkolivegreen', linewidth=2, label='Median')
    
    ##--Format x-axis on a log scale--##
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    
    ##--Format y-axis to leave extra space at the bottom--##
    plt.ylim(-100, 1500)
    
    plt.xticks([3, 10, 150, 1000, 3000], fontsize=20)
    plt.yticks(fontsize=16)
               
    plt.xlabel('Dp (nm)', fontsize=20)
    plt.ylabel('dN/dlogDp', fontsize=20)
    plt.title(f"FIRE-ACE Particle Number Size Distribution - {flight.replace('Flight', 'Flight ')} ({flight_date})")
    plt.legend(fontsize=18)
    
    plt.show()
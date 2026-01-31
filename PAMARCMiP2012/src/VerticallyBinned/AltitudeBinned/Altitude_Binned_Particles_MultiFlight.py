# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 13:43:38 2026

@author: repooley
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from datetime import date
import icartt

##################
##--Open Files--##
##################

##--Set the base directory to project folder--##
directory = r"C:\Users\repooley\REP_PhD\Arctic_NPF\PAMARCMiP2012\data\raw"

CPC10_R1 = icartt.Dataset(r'C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data\raw\CPC_R1\CPC3772_Polar6_20150408_R1_L2.ict')

##--Select flight (Flight1 thru Flight9)--##
flights_to_analyze = ["Flight1", "Flight2", "Flight3", "Flight6", 
                      "Flight7", "Flight8", "Flight9"]

##--Define function that creates datasets from filenames--##
def find_files(directory, flight, partial_name):
    flight_dir = os.path.join(directory, flight)
    search_pattern = os.path.join(flight_dir, f"*{partial_name}*")
    return sorted(glob.glob(search_pattern))

##--Store processed data here: --##
particle_dfs = []

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
    n_10_60 = pd.DataFrame({'35.5': n_10_60, 'time':time})
    
    #############################
    ##--Calculate Uncertainty--##        
    #############################
    
    ##--Pull CPC data from R1 data--##
    CPC10_R1_conc = CPC10_R1.data['conc']
    
    ##--Isolate zero periods, setting conservative upper limit of 50 counts--##
    ##--Numpy doesn't recognize -9999 as NaN, tell it to ignore these values--##
    CPC10_zeros_c = CPC10_R1_conc[(CPC10_R1_conc < 50) & (CPC10_R1_conc != -99999)]
    
    ##--Calculate standard deviation of zeros--##
    # Use ddof=1 for sample standard deviation
    CPC10_sigma = np.std(CPC10_zeros_c, ddof=1)
    
    greater10nm_error = 3*CPC10_sigma
    
    ##--This is the 75th percentile median uncertainty across NETCARE--##
    nuc_error_3sigma = 133.71 
    
    ##--UHSAS doesn't have zero periods, using Poisson counting uncertainty--##
    UHSAS_total_sqrt = np.sqrt(UHSAS_denorm_counts)
    
    ##--Use simple sum of UHSAS uncertainties per bin for conservative estimate--##
    ##--Similar result as using sqrt of squares but erring on side of caution--##
    UHSAS_total_error = UHSAS_total_sqrt.sum(axis=1)
    
    ##--Calculate error in difference between CPC10 and UHSAS + OPC--##
    aitken_error_3sigma = (((greater10nm_error)**2 + (UHSAS_total_error)**2)**(0.5))*3
    
    ##--Creates a Pandas dataframe for particle data--##
    df = pd.DataFrame({'Altitude': altitude, 'CPC10_conc': CPC10_filtered, 'UHSAS_total': UHSAS_total['Total_count'],
                       'n_10_60': n_10_60['35.5'], 'aitken_error_3sigma': aitken_error_3sigma})
    
    particle_dfs.append(df)

################
##--PLOTTING--##
################

##--NUCLEATING PARTICLES--##

num_bins = 128
all_alt = pd.concat([df["Altitude"] for df in particle_dfs])
min_alt = all_alt.min(skipna=True)
max_alt = all_alt.max(skipna=True)
bin_edges = np.linspace(min_alt, max_alt, num_bins + 1)

fig, axs = plt.subplots(1, 3, figsize=(9, 6), sharey=True)

cmap = plt.cm.viridis
n_flights = len(flights_to_analyze)
colors = [cmap(i / (n_flights - 1)) for i in range(n_flights)]

for i, flight in enumerate(flights_to_analyze):
    particle_df = particle_dfs[i].copy()

    particle_df['Alt_bin'] = pd.cut(particle_df['Altitude'], bins=bin_edges)

    ##--Binning--##
    binned_df = particle_df.groupby('Alt_bin', observed=False).agg(
        Alt_center=('Altitude', 'median'),
        CPC10_conc_center=('CPC10_conc', 'median'),
        UHSAS_total_center=('UHSAS_total', 'median'),
        n_10_60_center=('n_10_60', 'median')
    ).reset_index()
    
    color = colors[i]

    ##--UHSAS total (60+ nm)--##
    axs[0].plot(binned_df["UHSAS_total_center"], binned_df["Alt_center"],
                label=flight, color=color)

    ##--CPC 10--##
    axs[1].plot(binned_df["CPC10_conc_center"], binned_df["Alt_center"],
                label=flight, color=color)

    ##--Aitken--##
    axs[2].plot(binned_df["n_10_60_center"], binned_df["Alt_center"],
                label=flight, color=color)
    

axs[0].set_ylabel("Altitude (m)", fontsize=16)
axs[0].set_xlabel("Counts/cm³", fontsize=14)
axs[0].set_title("N ≥ 60 nm", fontsize=16)
#axs[0].set_xlim(-50, 2500)
axs[0].tick_params(axis='both', labelsize=11)

axs[1].set_title("N ≥ 10 nm", fontsize=16)
axs[1].set_xlabel("Counts/cm³", fontsize=14)
#axs[1].set_xlim(-50, 3500)
axs[1].tick_params(axis='both', labelsize=11)

axs[2].set_title("$N_{10-60}$", fontsize=16)
axs[2].set_xlabel("Counts/cm³", fontsize=14)
axs[2].tick_params(axis='both', labelsize=11)

axs[2].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=12)

plt.suptitle("PAMARCMiP 2012 Vertical Particle Profiles", fontsize=18)

plt.tight_layout(rect=[0, 0.05, 1, 0.99]) 

plt.show()
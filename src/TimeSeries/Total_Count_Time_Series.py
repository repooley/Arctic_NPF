# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 11:45:23 2025

@author: repooley
"""

import icartt
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
from datetime import datetime, timedelta

#########################
##--Open ICARTT Files--##
#########################

##--Set the base directory to project folder--##
directory = r"C:\Users\repooley\REP_PhD\NETCARE2015\data"

##--Select flight (Flight2 thru Flight10)--##
##--NO UHSAS FILES FOR FLIGHT1--##
flight = "Flight2"

##--Define function that creates datasets from filenames--##
def find_files(directory, flight, partial_name):
    ##--flight data are stored in a folder called "raw"--##
    flight_dir = os.path.join(directory, "raw", flight)
    search_pattern = os.path.join(flight_dir, f"*{partial_name}*")
    return sorted(glob.glob(search_pattern))

##--Meterological data from AIMMS monitoring system--##
aimms = icartt.Dataset(find_files(directory, flight, "AIMMS_POLAR6")[0])

##--UHSAS data--##
UHSAS = icartt.Dataset(find_files(directory, flight, 'UHSAS')[0])

##--OPC data--##
OPC = icartt.Dataset(find_files(directory, flight, 'OPC')[0])

##--CPC data--##
CPC10 = icartt.Dataset(find_files(directory, flight, 'CPC3772')[0])
CPC3 = icartt.Dataset(find_files(directory, flight, 'CPC3776')[0])

#########################
##--Pull & align data--##
#########################

##--AIMMS Data--##
altitude = aimms.data['Alt'] # in m
latitude = aimms.data['Lat'] # in degrees
temperature = aimms.data['Temp'] + 273.15 # in K
pressure = aimms.data['BP'] # in pa
aimms_time =aimms.data['TimeWave'] # seconds since midnight

##--USHAS Data--##
UHSAS_time = UHSAS.data['time'] # seconds since midnight
##--Total count is computed for N > 85 nm--##
UHSAS_total_num = UHSAS.data['total_number_conc'] # particles/cm^3

##--Bin data are in a CSV file--##
UHSAS_bins = pd.read_csv(r"C:\Users\repooley\REP_PhD\NETCARE2015\data\raw\NETCARE2015_UHSAS_bins.csv")

##--Make list of columns to pull, each named bin_x--##
##--Bins 1-13 not trustworthy. Bins 76-99 overlap with OPC, discard--##
##--Trim to use bins 14-76 (500>85 nm)--##
UHSAS_bin_num = [f'bin_{i}' for i in range(14, 75)]

##--Information for bins 14 thru 99--##
UHSAS_bin_center = UHSAS_bins['bin_avg'].iloc[14:75]
UHSAS_lower_bound = UHSAS_bins['lower_bound'].iloc[14:75]
UHSAS_upper_bound = UHSAS_bins['upper_bound'].iloc[14:75]

##--Put column names and content in a dictionary and then convert to a Pandas df--##
UHSAS_bins = pd.DataFrame({col: UHSAS.data[col] for col in UHSAS_bin_num})

##--Create new column names by rounding the bin center values to the nearest integer--##
UHSAS_new_col_names = UHSAS_bin_center.round().astype(int).tolist()

##--Rename the UHSAS_bins df columns to bin average values--##
UHSAS_bins.columns = UHSAS_new_col_names

##--Add time, total_num to UHSAS_bins df--##
UHSAS_bins.insert(0, 'Time', UHSAS_time)

##--Align UHSAS_bins time to AIMMS time--##
UHSAS_bins_aligned = UHSAS_bins.set_index('Time').reindex(aimms_time)

##--OPC Data--##
OPC_time = OPC.data['Time_UTC'] # seconds since midnight

##--Bin data are in a CSV file--##
OPC_bin_info = pd.read_csv(r"C:\Users\repooley\REP_PhD\NETCARE2015\data\raw\NETCARE2015_OPC_bins.csv")

##--Select bins greater than 500 nm (Channel 7 and greater)--##
OPC_bin_center = OPC_bin_info['bin_avg'].iloc[6:31]
OPC_lower_bound = OPC_bin_info['lower_bound'].iloc[6:31]
OPC_upper_bound = OPC_bin_info['upper_bound'].iloc[6:31]

##--Make list of columns to pull, each named Channel_x--##
OPC_bin_num = [f'Channel_{i}' for i in range(7, 32)]

##--Put column names and content in a dictionary and then convert to a Pandas df--##
OPC_bins = pd.DataFrame({col: OPC.data[col] for col in OPC_bin_num})

##--Create new column names by rounding the bin center values to the nearest integer--##
OPC_new_col_names = OPC_bin_center.round().astype(int).tolist()

##--Rename the OPC_bins df columns to bin average values--##
OPC_bins.columns = OPC_new_col_names

##--Add time, total_num to OPC_bins df--##
OPC_bins.insert(0, 'Time', OPC_time)

##--Align OPC_bins time to AIMMS time--##
OPC_bins_aligned = OPC_bins.set_index('Time').reindex(aimms_time)

##--10 nm CPC data--##
CPC10_time = CPC10.data['time']
CPC10_conc = CPC10.data['conc'] # count/cm^3

##--2.5 nm CPC data--##
CPC3_time = CPC3.data['time']
CPC3_conc = CPC3.data['conc'] # count/cm^3

##--Make CPC3 df and set index to CPC3 time--##
CPC3_df = pd.DataFrame({'time': CPC3_time, 'conc': CPC3_conc}).set_index('time')
##--Make a new df reindexed to aimms_time. Populate with CPC3 conc--##
CPC3_conc_aligned = CPC3_df.reindex(aimms_time)['conc']

##--Make CPC10 df and set index to CPC10 time--##
CPC10_df = pd.DataFrame({'time': CPC10_time, 'conc': CPC10_conc}).set_index('time')
##--Make a new df reindexed to aimms_time. Populate with CPC10 conc--##
CPC10_conc_aligned = CPC10_df.reindex(aimms_time)['conc']

##########################
##--Normalize OPC Data--##
##########################

##--OPC samples every six seconds. Most rows are NaN--##
##--Forward-fill NaN values to propagate last valid reading--##
##--Limit forward filling to 5 NaN rows--##
OPC_bins_filled = OPC_bins_aligned.ffill(limit=5)

##--Calculate dlogDp for each bin in numpy array--##
dlogDp = np.log(OPC_upper_bound.values) - np.log(OPC_lower_bound.values)

##--Get only particle count data (excluding 'Time')--##
OPC_particle_counts = OPC_bins_filled.loc[:, OPC_new_col_names]

##--Normalize counts by dividing by dlogDp across all rows--##
OPC_dNdlogDp = OPC_bins_filled.divide(dlogDp, axis=1)

##--Convert to STP!--##
P_STP = 101325  # Pa
T_STP = 273.15  # K

##--Create empty list for OPC particles--##
OPC_conc_STP = []

for OPC, T, P in zip(OPC_dNdlogDp.values, temperature, pressure):
    if np.isnan(T) or np.isnan(P):
        ##--Append with NaN if any input is NaN--##
        OPC_conc_STP.append([np.nan]*len(OPC))
    else:
        ##--Perform conversion if all inputs are valid--##
        corrected_OPC = OPC * (P_STP / P) * (T / T_STP)
        OPC_conc_STP.append(corrected_OPC)

##--Convert back to DataFrame with same columns and index--##
OPC_conc_STP = pd.DataFrame(OPC_conc_STP, columns=OPC_dNdlogDp.columns, index=OPC_dNdlogDp.index)

######################
##--Calc N(2.5-10)--##
######################

##--Convert to STP--##
##--Create empty list for CPC3 particles--##
CPC3_conc_STP = []

for CPC3, T, P in zip(CPC3_conc_aligned, temperature, pressure):
    if np.isnan(CPC3) or np.isnan(T) or np.isnan(P):
        ##--Append with NaN if any input is NaN--##
        CPC3_conc_STP.append(np.nan)
    else:
        ##--Perform conversion if all inputs are valid--##
        CPC3_conversion = CPC3 * (P_STP / P) * (T / T_STP)
        CPC3_conc_STP.append(CPC3_conversion)
    
##--Create empty list for CPC10 particles--##
CPC10_conc_STP = []

for CPC10, T, P in zip(CPC10_conc_aligned, temperature, pressure):
    if np.isnan(CPC10) or np.isnan(T) or np.isnan(P):
        ##--Append with NaN if any input is NaN--##
        CPC10_conc_STP.append(np.nan)
    else:
        ##--Perform conversion if all inputs are valid--##
        CPC10_conversion = CPC10 * (P_STP / P) * (T / T_STP)
        CPC10_conc_STP.append(CPC10_conversion)

##--Creates a Pandas dataframe for CPC data--##
CPC_df = pd.DataFrame({'Altitude': altitude, 'Latitude': latitude, 'CPC3_conc':CPC3_conc_STP, 'CPC10_conc': CPC10_conc_STP})

##--Calculate N3-10 particles--##
nuc_particles = (CPC_df['CPC3_conc'] - CPC_df['CPC10_conc'])

##--Change calculated particle counts less than zero to NaN--##
nuc_particles = np.where(nuc_particles >= 0, nuc_particles, np.nan)

##--Put N(2.5-10) bin center in a df--##
n_3_10_center = pd.DataFrame([6.25]) # Mean of 2.5 and 10

##--Create a dataframe for N 2.5-10--##
n_3_10 = pd.DataFrame({'time': aimms_time, '6': nuc_particles}).set_index('time')

#####################
##--Calc N(10-89)--##
#####################

##--Create df with UHSAS total counts--##
UHSAS_total = pd.DataFrame({'Time': UHSAS_time, 'Total_count': UHSAS_total_num})

##--Reindex UHSAS_total df to AIMMS time--##
UHSAS_total_aligned = UHSAS_total.set_index('Time').reindex(aimms_time)

##--Create df with CPC10 counts and set index to time--##
CPC10_counts = pd.DataFrame({'Time':aimms_time, 'Counts':CPC10_conc_STP}).set_index('Time')

##--Calculate particles below UHSAS lower cutoff--##
n_10_89 = (CPC10_counts['Counts'] - UHSAS_total_aligned['Total_count'])

##--Change calculated particle counts less than zero to NaN--##
n_10_89 = np.where(n_10_89 >= 0, n_10_89, np.nan)

##--Put N(10-85) bin center in a df--##
n_10_89_center = pd.DataFrame([49.5])

##--Convert n_10_85 to a df--##
n_10_89 = pd.DataFrame({'49.5': n_10_89, 'time':aimms_time}).set_index('time')

################
##--Plotting--##
################

##--Concatenate bin centers and reindex--##
bin_centers = pd.concat([n_3_10_center, n_10_89_center, UHSAS_bin_center, OPC_bin_center], axis=0).reset_index(drop=True)

##--Concatenate bin edges--##
combined_bin_edges = np.concatenate([
    [2.5],      # start of first bin
    [10],       # upper edge of N(2.5-10), also lower of next
    [89.32],       # upper edge of N(10-89), also lower of next
    UHSAS_upper_bound.values,  # UHSAS bins continue from 85
    OPC_upper_bound.values     # OPC bins continue from last UHSAS
])

##--Convert seconds since midnight to date time--##
aimms_hhmm = []

for seconds in aimms_time:
    ##--Choose arbitary start date--##
    time_obj = (datetime(1900, 1, 1) + timedelta(seconds=seconds)).time()
    aimms_hhmm.append(time_obj)

##--Calculate time edges for each bin, pcmesh doesn't expect time objects!--##
time_step = aimms_time[1] - aimms_time[0]  
time_edges = np.append(aimms_time, aimms_time[-1] + time_step)  # length N + 1

##--Create df containing UHSAS and OPC columns--##
optical_bins_aligned = pd.concat([n_3_10, n_10_89, UHSAS_bins_aligned, OPC_conc_STP], axis=1)

##--Apply rolling average to all other particle data--##
optical_bins_smoothed = optical_bins_aligned.rolling(window=30, min_periods=1, center=True).mean()

##--Numpy array expected by pcolormesh--##
optical_conc = optical_bins_smoothed.to_numpy().T  

##--Use pcolormesh which is more flexible than imshow--## 
fig, ax1 = plt.subplots(figsize=(12, 8))
c = ax1.pcolormesh(time_edges, combined_bin_edges, optical_conc, shading='auto', cmap='viridis')

##--Grab ticks--##
tick_seconds = ax1.get_xticks()
##--Convert from seconds since midnight to HH:MM using an arbitary date for datetime--##
tick_labels = [(datetime(1900, 1, 1) + timedelta(seconds=s)).strftime("%H") for s in tick_seconds]
ax1.set_xticks(tick_seconds)
ax1.set_xticklabels(tick_labels)
ax1.set_xlabel("Hour", fontsize=14)

ax1.set_title(f"Particle Size Distribution - {flight.replace('Flight', 'Flight ')}", fontsize=20, pad=20)
##--Set axis limits to match data range--##
ax1.set_xlim(time_edges[0], time_edges[-1])
#ax1.set_ylim([bin_center.min(), bin_center.max()])
ax1.set_yscale('log')
custom_ticks = [10, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
##--Apply custom ticks--##
ax1.yaxis.set_major_locator(ticker.FixedLocator(custom_ticks))

##--Format y-axis as regular numbers--##
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
ax1.tick_params(axis='y', labelsize=14)
ax1.tick_params(axis='x', labelsize=14)

#ax1.set_xlabel('Time (seconds since midnight UTC)', fontsize=14)
ax1.set_ylabel('Log of Particle Diameter (nm)', fontsize=14)

##--Add a colorbar below the plot--##
cb = plt.colorbar(c, ax=ax1, location='bottom', pad=0.13, shrink=0.65)
cb.set_label(label='Normalized Particle Concentration (dN/dlogDp) [scm⁻³]', fontsize=14)
cb.ax.tick_params(labelsize=14)

plt.tight_layout()
plt.show()

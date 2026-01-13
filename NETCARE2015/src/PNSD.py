# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 08:19:49 2025

@author: repooley
"""

import icartt
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.ticker as mticker

#########################
##--Open ICARTT Files--##
#########################

##--Set the base directory to project folder--##
directory = r"C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data"

##--Select flight (Flight2 thru Flight10)--##
##--NO UHSAS FILES FOR FLIGHT1--##
flight = "Flight2"

##--Above dome?--##
above_dome = True

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
UHSAS_bins = pd.read_csv(r"C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data\raw\NETCARE2015_UHSAS_bins.csv")

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
OPC_bin_info = pd.read_csv(r"C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data\raw\NETCARE2015_OPC_bins.csv")

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
    
PTemp_series = pd.Series(potential_temp, index=aimms_time)

##--Boolean mask for 285 K--##
mask = PTemp_series > 285

##--Mask UHSAS data--##
UHSAS_bins_aligned_mask = UHSAS_bins_aligned[mask]

##########################
##--Normalize OPC Data--##
##########################

##--OPC samples every six seconds. Most rows are NaN--##
##--Forward-fill NaN values to propagate last valid reading--##
##--Limit forward filling to 5 NaN rows--##
OPC_bins_filled = OPC_bins_aligned.ffill(limit=5)

##--Get only particle count data (excluding 'Time')--##
particle_cols = OPC_new_col_names

##--Calculate dlogDp for each bin in numpy array--##
dlogDp = np.log(OPC_upper_bound.values) - np.log(OPC_lower_bound.values)

##--Normalize--##
OPC_dNdlogDp = OPC_bins_filled[particle_cols].divide(dlogDp, axis=1)

##--Restore index--##
OPC_dNdlogDp.index = OPC_bins_filled.index

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

OPC_conc_STP_masked = OPC_conc_STP[mask]

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

##--Flatten n_3_10_center and change into a series--##
n_3_10_center = pd.Series(n_3_10_center.values.flatten())

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

##--Flatten--##
n_10_89_center = pd.Series(n_10_89_center.values.flatten())

##--Convert n_10_85 to a df--##
n_10_89 = pd.DataFrame({'49.5': n_10_89, 'time':aimms_time}).set_index('time')

###################################
##--Normalize nuc and grow bins--##
###################################

##--Calculate dlogDp--##
dlog_3_10 = np.log(10.0) - np.log(2.5)    # for 2.5 - 10 nm
dlog_10_89 = np.log(89.0) - np.log(10.0)  # for 10 - 89 nm 

##--Create dN/dlogDp--##
n_3_10_dNdlogDp = n_3_10['6'] / dlog_3_10
n_10_89_dNdlogDp = n_10_89['49.5'] / dlog_10_89

##--Mask n_3_10 to above the polar dome--##
n_3_10_masked = n_3_10_dNdlogDp[mask]

##--Mask n_10_89--##
n_10_89_masked = n_10_89_dNdlogDp[mask]

################
##--Plotting--##
################

##--Concatenate bin centers and reindex--##
bin_centers = pd.concat([n_3_10_center, n_10_89_center, UHSAS_bin_center, OPC_bin_center], axis=0).reset_index(drop=True)

##--Concatenate bin edges--##
combined_bin_edges_optical = np.concatenate([
    UHSAS_upper_bound.values,  # UHSAS bins continue from 85
   OPC_upper_bound.values     # OPC bins continue from last UHSAS
])

if above_dome: 

    ##--Create df containing UHSAS and OPC columns--##
    optical_bins_aligned = pd.concat([UHSAS_bins_aligned_mask, OPC_conc_STP_masked], axis=1)
else:
    ##--Create df containing UHSAS and OPC columns--##
    optical_bins_aligned = pd.concat([UHSAS_bins_aligned, OPC_conc_STP], axis=1)

##--Apply rolling average to particle data--##
optical_bins_smoothed = optical_bins_aligned.rolling(window=30, min_periods=1, center=True).mean()

##--Compute the median, 75th, and 90th percentiles of data--##
##--Apply smoothing--##
optical_bins_median = optical_bins_aligned.median(axis=0).rolling(window=30, min_periods=1, center=True).mean()
optical_bins_75th = optical_bins_aligned.quantile(q=0.75, axis=0).rolling(window=30, min_periods=1, center=True).mean()
optical_bins_25th = optical_bins_aligned.quantile(q=0.25, axis=0).rolling(window=30, min_periods=1, center=True).mean()
optical_bins_max = optical_bins_aligned.max(axis=0).rolling(window=30, min_periods=1, center=True).mean()
optical_bins_min = optical_bins_aligned.min(axis=0).rolling(window=30, min_periods=1, center=True).mean()

if above_dome: 
    n_3_10_median = n_3_10_masked.median()
    n_3_10_75th = n_3_10_masked.quantile(q=0.75)
    n_3_10_25th = n_3_10_masked.quantile(q=0.25)
    n_3_10_max = n_3_10_masked.max()
    n_3_10_min = n_3_10_masked.min()
    
    n_10_89_median = n_10_89_masked.median()
    n_10_89_75th = n_10_89_masked.quantile(q=0.75)
    n_10_89_25th = n_10_89_masked.quantile(q=0.25)
    n_10_89_max = n_10_89_masked.max()
    n_10_89_min = n_10_89_masked.min()
else: 
    n_3_10_median = n_3_10_dNdlogDp.median()
    n_3_10_75th = n_3_10_dNdlogDp.quantile(q=0.75)
    n_3_10_25th = n_3_10_dNdlogDp.quantile(q=0.25)
    n_3_10_max = n_3_10_dNdlogDp.max()
    n_3_10_min = n_3_10_dNdlogDp.min()
    
    n_10_89_median = n_10_89_dNdlogDp.median()
    n_10_89_75th = n_10_89_dNdlogDp.quantile(q=0.75)
    n_10_89_25th = n_10_89_dNdlogDp.quantile(q=0.25)
    n_10_89_max = n_10_89_dNdlogDp.max()
    n_10_89_min = n_10_89_dNdlogDp.min()

##--Set up figure--##
fig, ax = plt.subplots(1, 1, figsize=(12,6))

##--Add percentile ranges--##
ax.fill_between(combined_bin_edges_optical, optical_bins_min, optical_bins_max, 
                color='cadetblue', alpha=0.4, edgecolor='none')
ax.fill_between(combined_bin_edges_optical, optical_bins_25th, optical_bins_75th, 
                color='cadetblue', alpha=1, edgecolor='none')

##--y-axis fill between--##
ax.fill_between(n_3_10_center, n_3_10_min, n_3_10_max, color='cadetblue', 
                alpha=0.4, label='Full Range', edgecolor='none')
ax.fill_between(n_3_10_center, n_3_10_25th, n_3_10_75th, color='cadetblue', 
                alpha=1, label='Interquartile Range')


ax.fill_between(n_10_89_center, n_10_89_min, n_10_89_max, color='cadetblue', alpha=0.4, linewidth=3, edgecolor='none')
ax.fill_between(n_10_89_center, n_10_89_25th, n_10_89_75th, color='cadetblue', alpha=1, linewidth=3)

ax.vlines(x=2.5, ymin=-250, ymax=4000, colors='darkgrey', linewidth=1.5, linestyle='--')
ax.vlines(x=10, ymin=-250, ymax=4000, colors='darkgrey', linewidth=1.5, linestyle='--')
ax.vlines(x=89, ymin=-250, ymax=4000, colors='darkgrey', linewidth=1.5, linestyle='--')

##--Fill between on x-axis to give appearance of a full bin--##
##--Define bin edges--##
bin_edges = np.array([2.5, 10.0])
bin_edges2 = np.array([10, 89])

##--Repeat y-values across the bin--##
nuc_min_fill = np.array([n_3_10_min, n_3_10_min])
nuc_max_fill = np.array([n_3_10_max, n_3_10_max])
nuc_25_fill = np.array([n_3_10_25th, n_3_10_25th])
nuc_75_fill = np.array([n_3_10_75th, n_3_10_75th])

grow_min_fill = np.array([n_10_89_min, n_10_89_min])
grow_max_fill = np.array([n_10_89_max, n_10_89_max])
grow_25_fill = np.array([n_10_89_25th, n_10_89_25th])
grow_75_fill = np.array([n_10_89_75th, n_10_89_75th])

##--Fill full range--##
ax.fill_between(bin_edges, nuc_min_fill, nuc_max_fill, color='cadetblue', alpha=0.4, linewidth=3, edgecolor='none')
ax.fill_between(bin_edges2, grow_min_fill, grow_max_fill, color='cadetblue', alpha=0.4, linewidth=3, edgecolor='none')

##--Fill interquartile range--##
ax.fill_between(bin_edges, nuc_25_fill, nuc_75_fill, color='cadetblue', alpha=1, linewidth=3, edgecolor='none')
ax.fill_between(bin_edges2, grow_25_fill, grow_75_fill, color='cadetblue', alpha=1, linewidth=3, edgecolor='none')

##--Add medians--##
ax.plot(combined_bin_edges_optical, optical_bins_median, c='darkslategrey', linewidth=2, label='Median')

##--Fill n_3_10_median to edges--##
median_fill = np.array([n_3_10_median, n_3_10_median])
ax.plot(bin_edges, median_fill, c='darkslategrey', linewidth=2)

##--Same for n_10_89--##
median_fill2 = np.array([n_10_89_median, n_10_89_median])
ax.plot(bin_edges2, median_fill2, c='darkslategrey', linewidth=2)

##--Format x-axis on a log scale--##
ax.set_xscale('log')
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())

##--Format y-axis to leave extra space at the bottom--##
plt.ylim(-100, 1500)

plt.xticks([3, 10, 89, 1000, 10000], fontsize=16)
plt.yticks(fontsize=16)
           
plt.xlabel('Dp (nm)', fontsize=20)
plt.ylabel('dN/dlogDp', fontsize=20)
plt.title("NETCARE Particle Number Size Distribution", fontsize=20)
plt.legend(fontsize=18)

plt.show()
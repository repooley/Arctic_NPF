# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 16:56:06 2025

@author: repooley
"""

import icartt
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from scipy.stats import binned_statistic_2d

#########################
##--Open ICARTT Files--##
#########################

##--Set the base directory to project folder--##
directory = r"C:\Users\repooley\REP_PhD\NETCARE2015\data"

##--Select flight (Flight1 thru Flight10)--##
flight = "Flight10" # Flight1 AIMMS file currently broken at line 13234

##--Define function that creates datasets from filenames--##
def find_files(directory, flight, partial_name):
    ##--flight data are stored in a folder called "raw"--##
    flight_dir = os.path.join(directory, "raw", flight)
    search_pattern = os.path.join(flight_dir, f"*{partial_name}*")
    return sorted(glob.glob(search_pattern))

##--Meterological data from AIMMS monitoring system--##
aimms = icartt.Dataset(find_files(directory, flight, "AIMMS_POLAR6")[0])

##--CPC data--##
CPC10 = icartt.Dataset(find_files(directory, flight, 'CPC3772')[0])
CPC3 = icartt.Dataset(find_files(directory, flight, 'CPC3776')[0])

##--Trace gases--##
CO = icartt.Dataset(find_files(directory, flight, "CO_POLAR6")[0])
CO2 = icartt.Dataset(find_files(directory, flight, "CO2_POLAR6")[0])

##--Flight 2 has multiple ozone files requiring special handling--##
O3_files = find_files(directory, flight, "O3_")
if len(O3_files) == 0:
    raise FileNotFoundError("No O3 files found.")
elif len(O3_files) == 1 or flight != "Flight2": 
    O3 = icartt.Dataset(O3_files[0])
    O3_2 = None
##--Special case for Flight 2--##
else: 
    O3 = icartt.Dataset(O3_files[0])
    O3_2 = icartt.Dataset(O3_files[1])

#################
##--Pull data--##
#################

##--AIMMS Data--##
altitude = aimms.data['Alt'] # in m
latitude = aimms.data['Lat'] # in degrees
aimms_time =aimms.data['TimeWave'] # in seconds since midnight
temperature = aimms.data['Temp'] + 273.15 #in K
pressure = aimms.data['BP'] #in pa

##--Trace Gas Data--##
CO_conc = CO.data['CO_ppbv']
CO2_conc = CO2.data['CO2_ppmv']

##--Put O3 data in list to make concatenation easier--##
O3_starttime = list(O3.data['Start_UTC'])
O3_conc = list(O3.data['O3'])

##--Check for O3_2 data and stich to end of O3 if populated--##
if O3_2 is not None:
    O3_starttime += list(O3_2.data['Start_UTC'])
    O3_conc += list(O3_2.data['O3'])

##--10 nm CPC data--##
CPC10_time = CPC10.data['time']
CPC10_conc = CPC10.data['conc'] # count/cm^3

##--2.5 nm CPC data--##
CPC3_time = CPC3.data['time']
CPC3_conc = CPC3.data['conc'] # count/cm^3
    
##################
##--Align time--##
##################

##--Arbitary reference date for datetime conversion--##
reference_date = pd.to_datetime('2015-01-01')

##--O3 data: addressing different data resolution compared to AIMMS--##

##--Convert O3_starttime to a datetime object--##
O3_starttime_dt = pd.to_datetime(O3_starttime, unit='s', origin=reference_date)

##--Calculate the seconds since midnight--##
O3_seconds_since_midnight = O3_starttime_dt.hour * 3600 + O3_starttime_dt.minute * 60 + O3_starttime_dt.second

##--Create O3 dataframe--##
O3_df = pd.DataFrame({'Time_UTC': O3_seconds_since_midnight,'O3': O3_conc})

##--Reindex O3 data to AIMMS time and set non-overlapping time values to NaN--##
O3_aligned = O3_df.set_index('Time_UTC').reindex(aimms_time)
O3_aligned['O3'] = O3_aligned['O3'].where(O3_aligned.index.isin(aimms_time), np.nan)
O3_conc_aligned = O3_aligned['O3']

##--Other trace gas data: addressing different start/stop times than AIMMS--##
aimms_start = aimms_time.min()
aimms_end = aimms_time.max()

##--Handle CO data with different start/stop times than AIMMS--##
CO_time = CO.data['Time_UTC']

##--Trim CO data if it starts before AIMMS--##
if CO_time.min() < aimms_start:
    mask_start = CO_time >= aimms_start
    CO_time = CO_time[mask_start]
    CO_conc = CO_conc[mask_start]
    
##--Append CO data with NaNs if it ends before AIMMS--##
if CO_time.max() < aimms_end: 
    missing_times = np.arange(CO_time.max()+1, aimms_end +1)
    CO_time = np.concatenate([CO_time, missing_times])
    CO_conc = np.concatenate([CO_conc, [np.nan]*len(missing_times)])

##--Create a DataFrame for CO data and reindex to AIMMS time, setting non-overlapping times to nan--##
CO_df = pd.DataFrame({'Time_UTC': CO_time, 'CO_ppbv': CO_conc})
CO_aligned = CO_df.set_index('Time_UTC').reindex(aimms_time)
CO_aligned['CO_ppbv']= CO_aligned['CO_ppbv'].where(CO_aligned.index.isin(aimms_time), np.nan)
CO_conc_aligned = CO_aligned['CO_ppbv']

##--Handle CO2 data with different start/stop times than AIMMS--##
CO2_time = CO2.data['Time_UTC']

##--Trim CO2 data if it starts before AIMMS--##
if CO2_time.min() < aimms_start:
    mask_start = CO2_time >= aimms_start
    CO2_time = CO2_time[mask_start]
    CO2_conc = CO2_conc[mask_start]
    
##--Append CO2 data with NaNs if it ends before AIMMS--##
if CO2_time.max() < aimms_end: 
    missing_times = np.arange(CO2_time.max()+1, aimms_end +1)
    CO2_time = np.concatenate([CO2_time, missing_times])
    CO2_conc = np.concatenate([CO2_conc, [np.nan]*len(missing_times)])

##--Create a DataFrame for CO2 data and reindex to AIMMS time, setting non-overlapping times to nan--##
CO2_df = pd.DataFrame({'Time_UTC': CO2_time, 'CO2_ppmv': CO2_conc})
CO2_aligned = CO2_df.set_index('Time_UTC').reindex(aimms_time)
CO2_aligned['CO2_ppmv']=CO2_aligned['CO2_ppmv'].where(CO2_aligned.index.isin(aimms_time), np.nan)
CO2_conc_aligned = CO2_aligned['CO2_ppmv']

##--Trim CPC3 data if it starts before AIMMS--##
if CPC3_time.min() < aimms_start:
    mask_start = CPC3_time >= aimms_start
    CPC3_time = CPC3_time[mask_start]
    CPC3_conc = CPC3_conc[mask_start]
    
##--Append CPC3 data with NaNs if it ends before AIMMS--##
if CPC3_time.max() < aimms_end: 
    missing_times = np.arange(CPC3_time.max()+1, aimms_end +1)
    CPC3_time = np.concatenate([CPC3_time, missing_times])
    CPC3_conc = np.concatenate([CPC3_conc, [np.nan]*len(missing_times)])

##--Create a DataFrame for CPC3 data and reindex to AIMMS time, setting non-overlapping times to nan--##
CPC3_df = pd.DataFrame({'time': CPC3_time, 'conc': CPC3_conc})
CPC3_aligned = CPC3_df.set_index('time').reindex(aimms_time)
CPC3_aligned['conc']=CPC3_aligned['conc'].where(CPC3_aligned.index.isin(aimms_time), np.nan)
CPC3_conc_aligned = CPC3_aligned['conc']

##--Handle CPC10 data with different start/stop times than AIMMS--##
CPC10_time = CPC10.data['time']

##--Trim CPC10 data if it starts before AIMMS--##
if CPC10_time.min() < aimms_start:
    mask_start = CPC10_time >= aimms_start
    CPC10_time = CPC10_time[mask_start]
    CPC10_conc = CPC10_conc[mask_start]
    
##--Append CPC10 data with NaNs if it ends before AIMMS--##
if CPC10_time.max() < aimms_end: 
    missing_times = np.arange(CPC10_time.max()+1, aimms_end +1)
    CPC10_time = np.concatenate([CPC10_time, missing_times])
    CPC10_conc = np.concatenate([CPC10_conc, [np.nan]*len(missing_times)])

##--Create a DataFrame for CPC10 data and reindex to AIMMS time, setting non-overlapping times to nan--##
CPC10_df = pd.DataFrame({'time': CPC10_time, 'conc': CPC10_conc})
CPC10_aligned = CPC10_df.set_index('time').reindex(aimms_time)
CPC10_aligned['conc']=CPC10_aligned['conc'].where(CPC10_aligned.index.isin(aimms_time), np.nan)
CPC10_conc_aligned = CPC10_aligned['conc']

######################
##--Convert to STP--##
######################

P_STP = 101325  # Pa
T_STP = 273.15  # K

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

#######################################
##--Calculate potential temperature--##
#######################################

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
##--Create 2D histogram--##
###########################

##--Float type NaNs in potential_temp cannot convert to int, so must be removed--##
CPC3_df = pd.DataFrame({'PTemp': potential_temp, 'Latitude': latitude, 'CPC3':CPC3_conc_STP})
CPC3_clean_df = CPC3_df.dropna()

##--Make separate df for CPC10 and N3-10 to preserve as much data as possible--##
CPC10_df = pd.DataFrame({'PTemp': potential_temp, 'Latitude': latitude, 'CPC10': CPC10_conc_STP})
CPC10_clean_df = CPC10_df.dropna()

##--Calculate N3-10 particles--##
nuc_particles = (CPC3_df['CPC3'] - CPC10_df['CPC10'])

##--Change calculated particle counts less than zero to NaN--##
nuc_particles = np.where(nuc_particles >= 0, nuc_particles, np.nan)

nuc_df = pd.DataFrame({'PTemp': potential_temp, 'Latitude': latitude, 'nuc_particles': nuc_particles})

##--Creates separate dfs to preserve data--##
##--Including nuc_particles downsizes dataset to instances of N3-10. Comment out if full dataset desired--##
O3_df = pd.DataFrame({'PTemp': potential_temp, 'Latitude': latitude, 'O3_conc': O3_conc_aligned, 'nuc_particles': nuc_particles})
CO_df = pd.DataFrame({'PTemp': potential_temp, 'Latitude': latitude, 'CO_conc': CO_conc_aligned, 'nuc_particles': nuc_particles})
CO2_df = pd.DataFrame({'PTemp': potential_temp, 'Latitude': latitude, 'CO2_conc': CO2_conc_aligned, 'nuc_particles': nuc_particles})

##--Drop NaNs to prevent issues with potential_temp floats--##
clean_O3_df = O3_df.dropna()
clean_CO_df = CO_df.dropna()
clean_CO2_df = CO2_df.dropna()

##--Define number of bins--##
num_bins_lat = 4
num_bins_ptemp = 12

##--Compute global min/max values across all data BEFORE dropping NaNs--##
lat_min, lat_max = np.nanmin(latitude), np.nanmax(latitude)
ptemp_min, ptemp_max = np.nanmin(potential_temp), np.nanmax(potential_temp)

##--Generate common bin edges using specified number of bins--##
common_lat_bin_edges = np.linspace(lat_min, lat_max, num_bins_lat + 1)
common_ptemp_bin_edges = np.linspace(ptemp_min, ptemp_max, num_bins_ptemp + 1)

##--Make 2D histogram using common bins--##
CO_bin_medians, _, _, _ = binned_statistic_2d(
    clean_CO_df['Latitude'], clean_CO_df['PTemp'], clean_CO_df['CO_conc'], 
    statistic='median', bins=[common_lat_bin_edges, common_ptemp_bin_edges])

O3_bin_medians, _, _, _ = binned_statistic_2d(
    clean_O3_df['Latitude'], clean_O3_df['PTemp'], clean_O3_df['O3_conc'], 
    statistic='median', bins=[common_lat_bin_edges, common_ptemp_bin_edges])

CO2_bin_medians, _, _, _ = binned_statistic_2d(
    clean_CO2_df['Latitude'], clean_CO2_df['PTemp'], clean_CO2_df['CO2_conc'], 
    statistic='median', bins=[common_lat_bin_edges, common_ptemp_bin_edges])

################
##--PLOTTING--##
################

##--O3--##
fig1, ax1 = plt.subplots(figsize=(8, 6))

##--Make special color map where 0 values are white--##
new_cmap = plt.get_cmap('viridis')

##--Values under specified minimum will be white--##
new_cmap.set_under('w')

##--Use pcolormesh for the plot, set minimum value for viridis colors as 1--##
O3_plot = ax1.pcolormesh(common_lat_bin_edges, common_ptemp_bin_edges, O3_bin_medians.T,  # Transpose to align correctly
    shading='auto', cmap=new_cmap, vmin=0, vmax=80)

##--Add dashed horizontal lines for the polar dome boundaries--##
ax1.axhline(y=275, color='k', linestyle='--', linewidth=1)
ax1.axhline(y=299, color='k', linestyle='--', linewidth=1)

##--Add colorbar--##
cb = fig1.colorbar(O3_plot, ax=ax1)
cb.minorticks_on()
cb.ax.tick_params(labelsize=16)
cb.set_label('O\u2083 ppbv', fontsize=16)

# Set axis labels
ax1.set_xlabel('Latitude (°)', fontsize=16)
ax1.set_ylabel('Potential Temperature \u0398 (K)', fontsize=16)
ax1.tick_params(axis='both', labelsize=16)
ax1.set_title(f"O\u2083 Mixing Ratio - {flight.replace('Flight', 'Flight ')}", fontsize=18)
#ax1.set_ylim(244, 301)
#ax1.set_xlim(82.4, 83.4)

##--Base output path in directory--##
output_path = r"C:\Users\repooley\REP_PhD\NETCARE2015\data\processed\CurtainPlots\TraceGas\PTempLatitude"

##--Use f-string to save file with flight# appended--##
output_path = f"{output_path}\\{flight}_O3.png"
plt.savefig(output_path, dpi=600, bbox_inches='tight') 
plt.tight_layout()
plt.show()

##--CO--##
fig2, ax2 = plt.subplots(figsize=(8, 6))

##--Use pcolormesh for the plot, set minimum value for viridis colors as 1--##
CO_plot = ax2.pcolormesh(common_lat_bin_edges, common_ptemp_bin_edges, CO_bin_medians.T,  # Transpose to align correctly
    shading='auto', cmap=new_cmap, vmin=100, vmax=250)

##--Add dashed horizontal lines for the polar dome boundaries--##
ax2.axhline(y=275, color='k', linestyle='--', linewidth=1)
ax2.axhline(y=299, color='k', linestyle='--', linewidth=1)

##--Add colorbar--##
cb = fig2.colorbar(CO_plot, ax=ax2)
cb.minorticks_on()
cb.ax.tick_params(labelsize=16)
cb.set_label('CO ppbv', fontsize=16)

##--Set axis labels--##
ax2.set_xlabel('Latitude (°)', fontsize=16)
ax2.set_ylabel('Potential Temperature \u0398 (K)', fontsize=16)
ax2.tick_params(axis='both', labelsize=16)
ax2.set_title(f"CO Mixing Ratio - {flight.replace('Flight', 'Flight ')}", fontsize=18)
#ax2.set_ylim(244, 301)
#ax2.set_xlim(82.4, 83.4)

##--Base output path in directory--##
output_path = r"C:\Users\repooley\REP_PhD\NETCARE2015\data\processed\CurtainPlots\TraceGas\PTempLatitude"

##--Use f-string to save file with flight# appended--##
output_path = f"{output_path}\\{flight}_CO.png"
plt.savefig(output_path, dpi=600, bbox_inches='tight') 
plt.tight_layout()
plt.show()

##--CO2--##
fig3, ax3 = plt.subplots(figsize=(8, 6))

##--Use pcolormesh for the plot, set minimum value for viridis colors as 1--##
CO2_plot = ax3.pcolormesh(common_lat_bin_edges, common_ptemp_bin_edges, CO2_bin_medians.T,  # Transpose to align correctly
    shading='auto', cmap=new_cmap, vmin=400, vmax=410)

##--Add dashed horizontal lines for the polar dome boundaries--##
ax3.axhline(y=275, color='k', linestyle='--', linewidth=1)
ax3.axhline(y=299, color='k', linestyle='--', linewidth=1)

##--Add colorbar--##
cb = fig3.colorbar(CO2_plot, ax=ax3)
cb.minorticks_on()
cb.ax.tick_params(labelsize=16)
cb.set_label('CO\u2082 ppmv', fontsize=16)

# Set axis labels
ax3.set_xlabel('Latitude (°)', fontsize=16)
ax3.set_ylabel('Potential Temperature \u0398 (K)', fontsize=16)
ax3.tick_params(axis='both', labelsize=16)
ax3.set_title(f"CO\u2082 Mixing Ratio - {flight.replace('Flight', 'Flight ')}", fontsize=18)
#ax3.set_ylim(244, 301)
#ax3.set_xlim(82.4, 83.4)

##--Base output path in directory--##
output_path = r"C:\Users\repooley\REP_PhD\NETCARE2015\data\processed\CurtainPlots\TraceGas\PTempLatitude"

##--Use f-string to save file with flight# appended--##
output_path = f"{output_path}\\{flight}_CO2.png"
plt.savefig(output_path, dpi=600, bbox_inches='tight') 
plt.tight_layout()
plt.show()

########################
##--Diagnostic Plots--##
########################

##--Remove hashtags below to comment out this section--##
#'''

##--Counts per bin for O3 data--##
O3_bin_counts, _, _, _ = binned_statistic_2d(clean_O3_df['Latitude'], 
    clean_O3_df['PTemp'], clean_O3_df['O3_conc'], statistic='count', bins=[common_lat_bin_edges, common_ptemp_bin_edges])
 
##--Counts per bin for CPC10 data--##
CO_bin_counts, _, _, _ = binned_statistic_2d(clean_CO_df['Latitude'], 
    clean_CO_df['PTemp'], clean_CO_df['CO_conc'], statistic='count', bins=[common_lat_bin_edges, common_ptemp_bin_edges])
 
##--Counts per bin for N3-10 particles--##
CO2_bin_counts, _, _, _ = binned_statistic_2d(clean_CO2_df['Latitude'], 
    clean_CO2_df['PTemp'], clean_CO2_df['CO2_conc'], statistic='count', bins=[common_lat_bin_edges, common_ptemp_bin_edges])

##--Plotting--##

##--O3--##
fig1, ax1 = plt.subplots(figsize=(8, 6))

##--Make special color map where 0 values are white--##
new_cmap = plt.get_cmap('inferno')

##--Values under specified minimum will be white--##
new_cmap.set_under('w')

##--Use pcolormesh for the plot, set minimum value for viridis colors as 1--##
O3_plot = ax1.pcolormesh(common_lat_bin_edges, common_ptemp_bin_edges, O3_bin_counts.T,  # Transpose to align correctly
    shading='auto', cmap=new_cmap, vmin=1, vmax=150)

##--Add dashed horizontal lines for the polar dome boundaries--##
ax1.axhline(y=275, color='k', linestyle='--', linewidth=1)
ax1.axhline(y=299, color='k', linestyle='--', linewidth=1)

##--Add colorbar--##
cb = fig1.colorbar(O3_plot, ax=ax1)
cb.minorticks_on()
cb.ax.tick_params(labelsize=16)
cb.set_label('Number of Data Points', fontsize=16)

# Set axis labels
ax1.set_xlabel('Latitude (°)', fontsize=16)
ax1.set_ylabel('Potential Temperature \u0398 (K)', fontsize=16)
ax1.tick_params(axis='both', labelsize=16)
ax1.set_title(f"O\u2083 Counts per Bin - {flight.replace('Flight', 'Flight ')}", fontsize=18)
#ax1.set_ylim(238, 301)
#ax1.set_xlim(79.5, 83.7)

##--Base output path in directory--##
output_path = r"C:\Users\repooley\REP_PhD\NETCARE2015\data\processed\CurtainPlots\TraceGas\PTempLatitude"

##--Use f-string to save file with flight# appended--##
output_path = f"{output_path}\\{flight}_O3_diagnostic.png"
plt.savefig(output_path, dpi=600, bbox_inches='tight') 
plt.tight_layout()
plt.show()

##--CO--##
fig2, ax2 = plt.subplots(figsize=(8, 6))

##--Use pcolormesh for the plot, set minimum value for viridis colors as 1--##
CO_plot = ax2.pcolormesh(common_lat_bin_edges, common_ptemp_bin_edges, CO_bin_counts.T,  # Transpose to align correctly
    shading='auto', cmap=new_cmap, vmin=1, vmax=1500)

##--Add dashed horizontal lines for the polar dome boundaries--##
ax2.axhline(y=275, color='k', linestyle='--', linewidth=1)
ax2.axhline(y=299, color='k', linestyle='--', linewidth=1)

##--Add colorbar--##
cb = fig2.colorbar(CO_plot, ax=ax2)
cb.minorticks_on()
cb.ax.tick_params(labelsize=16)
cb.set_label('Number of Data Points', fontsize=16)

##--Set axis labels--##
ax2.set_xlabel('Latitude (°)', fontsize=16)
ax2.set_ylabel('Potential Temperature \u0398 (K)', fontsize=16)
ax2.tick_params(axis='both', labelsize=16)
ax2.set_title(f"CO Counts per Bin - {flight.replace('Flight', 'Flight ')}", fontsize=18)
#ax2.set_ylim(238, 301)
#ax2.set_xlim(79.5, 83.7)

##--Base output path in directory--##
output_path = r"C:\Users\repooley\REP_PhD\NETCARE2015\data\processed\CurtainPlots\TraceGas\PTempLatitude"

##--Use f-string to save file with flight# appended--##
output_path = f"{output_path}\\{flight}_CO_diagnostic.png"
plt.savefig(output_path, dpi=600, bbox_inches='tight') 
plt.tight_layout()
plt.show()

##--CO2--##
fig3, ax3 = plt.subplots(figsize=(8, 6))

##--Use pcolormesh for the plot, set minimum value for viridis colors as 1--##
CO2_plot = ax3.pcolormesh(common_lat_bin_edges, common_ptemp_bin_edges, CO2_bin_counts.T,  # Transpose to align correctly
    shading='auto', cmap=new_cmap, vmin=1, vmax=1500)

##--Add dashed horizontal lines for the polar dome boundaries--##
ax3.axhline(y=275, color='k', linestyle='--', linewidth=1)
ax3.axhline(y=299, color='k', linestyle='--', linewidth=1)

##--Add colorbar--##
cb = fig3.colorbar(CO2_plot, ax=ax3)
cb.minorticks_on()
cb.ax.tick_params(labelsize=16)
cb.set_label('Number of Data Points', fontsize=16)

# Set axis labels
ax3.set_xlabel('Latitude (°)', fontsize=16)
ax3.set_ylabel('Potential Temperature \u0398 (K)', fontsize=16)
ax3.tick_params(axis='both', labelsize=16)
ax3.set_title(f"CO\u2082 Counts per Bin - {flight.replace('Flight', 'Flight ')}", fontsize=18)
#ax3.set_ylim(238, 301)
#ax3.set_xlim(79.5, 83.7)

##--Base output path in directory--##
output_path = r"C:\Users\repooley\REP_PhD\NETCARE2015\data\processed\CurtainPlots\TraceGas\PTempLatitude"

##--Use f-string to save file with flight# appended--##
output_path = f"{output_path}\\{flight}_CO2_diagnostic.png"
plt.savefig(output_path, dpi=600, bbox_inches='tight') 
plt.tight_layout()
plt.show()

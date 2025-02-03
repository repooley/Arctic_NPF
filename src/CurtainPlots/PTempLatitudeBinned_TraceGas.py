# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 10:38:32 2025

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
flight = "Flight2" # Flight1 AIMMS file currently broken at line 13234

##--Define function that creates datasets from filenames--##
def find_files(directory, flight, partial_name):
    ##--flight data are stored in a folder called "raw"--##
    flight_dir = os.path.join(directory, "raw", flight)
    search_pattern = os.path.join(flight_dir, f"*{partial_name}*")
    return sorted(glob.glob(search_pattern))

##--Meterological data from AIMMS monitoring system--##
aimms = icartt.Dataset(find_files(directory, flight, "AIMMS_POLAR6")[0])

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
temperature = aimms.data['Temp'] #in C
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

####################
##--Calculations--##
####################

##--Convert absolute temperature to potential temperature--##
##--Constants--##
p_0 = 1E5 # Reference pressure in Pa (1000 hPa)
k = 0.286 # Poisson constant for dry air

##--Convert temperature from Celcius to Kelvin--##
temperature_k = np.array(temperature) + 273.15

##--Generate empty list for potential temperature output--##
potential_temp = []

##--Calculate potential temperature from ambient temp & pressure--##
for T, P in zip(temperature_k, pressure):
    p_t = T*(p_0/P)**k
    potential_temp.append(p_t)

###########################
##--Create 2D histogram--##
###########################

##--Creates separate dfs to preserve data--##
O3_df = pd.DataFrame({'PTemp': potential_temp, 'Latitude': latitude, 'O3_conc': O3_conc_aligned})
CO_df = pd.DataFrame({'PTemp': potential_temp, 'Latitude': latitude, 'CO_conc': CO_conc_aligned})
CO2_df = pd.DataFrame({'PTemp': potential_temp, 'Latitude': latitude, 'CO2_conc': CO2_conc_aligned})

##--Drop NaNs to prevent issues with potential_temp floats--##
clean_O3_df = O3_df.dropna()
clean_CO_df = CO_df.dropna()
clean_CO2_df = CO2_df.dropna()

##--Define number of bins here--##
num_bins = 30

##--Determine bin edges--##
O3_lat_bin_edges = np.linspace(clean_O3_df['Latitude'].min(), clean_O3_df['Latitude'].max(), num_bins + 1)
O3_ptemp_bin_edges = np.linspace(clean_O3_df['PTemp'].min(), clean_O3_df['PTemp'].max(), num_bins + 1)
CO_lat_bin_edges = np.linspace(clean_CO_df['Latitude'].min(), clean_CO_df['Latitude'].max(), num_bins +1)
CO_ptemp_bin_edges = np.linspace(clean_CO_df['PTemp'].min(), clean_CO_df['PTemp'].max(), num_bins +1)
CO2_lat_bin_edges = np.linspace(clean_CO2_df['Latitude'].min(), clean_CO2_df['Latitude'].max(), num_bins +1)
CO2_ptemp_bin_edges = np.linspace(clean_CO2_df['PTemp'].min(), clean_CO2_df['PTemp'].max(), num_bins +1)

##--Make 2d histogram and compute median RH in each bin--##
O3_bin_medians, O3_x_edges, O3_y_edges, _ = binned_statistic_2d(clean_O3_df['Latitude'], 
    clean_O3_df['PTemp'], clean_O3_df['O3_conc'], statistic='median', bins=[O3_lat_bin_edges, O3_ptemp_bin_edges])
CO_bin_medians, CO_x_edges, CO_y_edges, _ = binned_statistic_2d(clean_CO_df['Latitude'], 
    clean_CO_df['PTemp'], clean_CO_df['CO_conc'], statistic='median', bins=[CO_lat_bin_edges, CO_ptemp_bin_edges])
CO2_bin_medians, CO2_x_edges, CO2_y_edges, _ = binned_statistic_2d(clean_CO2_df['Latitude'], 
    clean_CO2_df['PTemp'], clean_CO2_df['CO2_conc'], statistic='median', bins=[CO2_lat_bin_edges, CO2_ptemp_bin_edges])

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
O3_plot = ax1.pcolormesh(O3_x_edges, O3_y_edges, O3_bin_medians.T,  # Transpose to align correctly
    shading='auto', cmap=new_cmap, vmin=0, vmax=80)

##--Add colorbar--##
cb = fig1.colorbar(O3_plot, ax=ax1)
cb.minorticks_on()
cb.set_label('O\u2083 ppbv', fontsize=12)

# Set axis labels
ax1.set_xlabel('Latitude (°)', fontsize=12)
ax1.set_ylabel('Potential Temperature \u0398 (K)', fontsize=12)
ax1.set_title(f"O\u2083 Mixing Ratio - {flight.replace('Flight', 'Flight ')}")

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
CO_plot = ax2.pcolormesh(CO_x_edges, CO_y_edges, CO_bin_medians.T,  # Transpose to align correctly
    shading='auto', cmap=new_cmap, vmin=100, vmax=250)

##--Add colorbar--##
cb = fig2.colorbar(CO_plot, ax=ax2)
cb.minorticks_on()
cb.set_label('CO ppbv', fontsize=12)

##--Set axis labels--##
ax2.set_xlabel('Latitude (°)', fontsize=12)
ax2.set_ylabel('Potential Temperature \u0398 (K)', fontsize=12)
ax2.set_title(f"CO Mixing Ratio - {flight.replace('Flight', 'Flight ')}")

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
CO2_plot = ax3.pcolormesh(CO2_x_edges, CO2_y_edges, CO2_bin_medians.T,  # Transpose to align correctly
    shading='auto', cmap=new_cmap, vmin=400, vmax=410)

##--Add colorbar--##
cb = fig3.colorbar(CO2_plot, ax=ax3)
cb.minorticks_on()
cb.set_label('CO\u2082 ppmv', fontsize=12)

# Set axis labels
ax3.set_xlabel('Latitude (°)', fontsize=12)
ax3.set_ylabel('Potential Temperature \u0398 (K)', fontsize=12)
ax3.set_title(f"CO\u2082 Mixing Ratio - {flight.replace('Flight', 'Flight ')}")

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
    clean_O3_df['PTemp'], clean_O3_df['O3_conc'], statistic='count', bins=[O3_lat_bin_edges, O3_ptemp_bin_edges])
 
##--Counts per bin for CPC10 data--##
CO_bin_counts, _, _, _ = binned_statistic_2d(clean_CO_df['Latitude'], 
    clean_CO_df['PTemp'], clean_CO_df['CO_conc'], statistic='count', bins=[CO_lat_bin_edges, CO_ptemp_bin_edges])
 
##--Counts per bin for N3-10 particles--##
CO2_bin_counts, _, _, _ = binned_statistic_2d(clean_CO2_df['Latitude'], 
    clean_CO2_df['PTemp'], clean_CO2_df['CO2_conc'], statistic='count', bins=[CO2_lat_bin_edges, CO2_ptemp_bin_edges])

##--Plotting--##

##--O3--##
fig1, ax1 = plt.subplots(figsize=(8, 6))

##--Make special color map where 0 values are white--##
new_cmap = plt.get_cmap('inferno')

##--Values under specified minimum will be white--##
new_cmap.set_under('w')

##--Use pcolormesh for the plot, set minimum value for viridis colors as 1--##
O3_plot = ax1.pcolormesh(O3_x_edges, O3_y_edges, O3_bin_counts.T,  # Transpose to align correctly
    shading='auto', cmap=new_cmap, vmin=1, vmax=80)

##--Add colorbar--##
cb = fig1.colorbar(O3_plot, ax=ax1)
cb.minorticks_on()
cb.set_label('Number of Data Points', fontsize=12)

# Set axis labels
ax1.set_xlabel('Latitude (°)', fontsize=12)
ax1.set_ylabel('Potential Temperature \u0398 (K)', fontsize=12)
ax1.set_title(f"O\u2083 Counts per Bin - {flight.replace('Flight', 'Flight ')}")

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
CO_plot = ax2.pcolormesh(CO_x_edges, CO_y_edges, CO_bin_counts.T,  # Transpose to align correctly
    shading='auto', cmap=new_cmap, vmin=1, vmax=750)

##--Add colorbar--##
cb = fig2.colorbar(CO_plot, ax=ax2)
cb.minorticks_on()
cb.set_label('Number of Data Points', fontsize=12)

##--Set axis labels--##
ax2.set_xlabel('Latitude (°)', fontsize=12)
ax2.set_ylabel('Potential Temperature \u0398 (K)', fontsize=12)
ax2.set_title(f"CO Counts per Bin - {flight.replace('Flight', 'Flight ')}")

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
CO2_plot = ax3.pcolormesh(CO2_x_edges, CO2_y_edges, CO2_bin_counts.T,  # Transpose to align correctly
    shading='auto', cmap=new_cmap, vmin=1, vmax=750)

##--Add colorbar--##
cb = fig3.colorbar(CO2_plot, ax=ax3)
cb.minorticks_on()
cb.set_label('Number of Data Points', fontsize=12)

# Set axis labels
ax3.set_xlabel('Latitude (°)', fontsize=12)
ax3.set_ylabel('Potential Temperature \u0398 (K)', fontsize=12)
ax3.set_title(f"CO\u2082 Counts per Bin - {flight.replace('Flight', 'Flight ')}")

##--Base output path in directory--##
output_path = r"C:\Users\repooley\REP_PhD\NETCARE2015\data\processed\CurtainPlots\TraceGas\PTempLatitude"

##--Use f-string to save file with flight# appended--##
output_path = f"{output_path}\\{flight}_CO2_diagnostic.png"
plt.savefig(output_path, dpi=600, bbox_inches='tight') 
plt.tight_layout()
plt.show()


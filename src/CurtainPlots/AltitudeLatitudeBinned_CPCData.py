# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 15:51:24 2025

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
flight = "Flight9" # Flight1 AIMMS file currently broken at line 13234

##--Define function that creates datasets from filenames--##
def find_files(directory, flight, partial_name):
    ##--flight data are stored in a folder called "raw"--##
    flight_dir = os.path.join(directory, "raw", flight)
    search_pattern = os.path.join(flight_dir, f"*{partial_name}*")
    return sorted(glob.glob(search_pattern))

##--Meterological data from AIMMS monitoring system--##
aimms = icartt.Dataset(find_files(directory, flight, "AIMMS_POLAR6")[0])

##--Not pulling properly for flight 6, unsure why--##
##--CPC data--##
CPC10 = icartt.Dataset(find_files(directory, flight, 'CPC3772')[0])
CPC3 = icartt.Dataset(find_files(directory, flight, 'CPC3776')[0])

#################
##--Pull data--##
#################

##--AIMMS Data--##
altitude = aimms.data['Alt'] # in m
latitude = aimms.data['Lat'] # in degrees
temperature = aimms.data['Temp'] + 273.15 # in K
pressure = aimms.data['BP'] # in pa
aimms_time =aimms.data['TimeWave'] # seconds since midnight

##--10 nm CPC data--##
CPC10_time = CPC10.data['time']
CPC10_conc = CPC10.data['conc'] # count/cm^3

##--2.5 nm CPC data--##
CPC3_time = CPC3.data['time']
CPC3_conc = CPC3.data['conc'] # count/cm^3

##################
##--Align data--##
##################

##--Establish AIMMS start/stop times--##
aimms_end = aimms_time.max()
aimms_start = aimms_time.min()

##--Handle CPC3 data with different start/stop times than AIMMS--##
CPC3_time = CPC3.data['time']

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

###########################
##--Create 2D histogram--##
###########################

##--Creates a Pandas dataframe for CPC data--##
CPC_df = pd.DataFrame({'Altitude': altitude, 'Latitude': latitude, 'CPC3_conc':CPC3_conc_STP, 'CPC10_conc': CPC10_conc_STP})

##--Calculate N3-10 particles--##
nuc_particles = (CPC_df['CPC3_conc'] - CPC_df['CPC10_conc'])

##--Change calculated particle counts less than zero to NaN--##
nuc_particles = np.where(nuc_particles >= 0, nuc_particles, np.nan)

##--Append the dataframe with nucleated particles--##
CPC_df['nuc_particles'] = nuc_particles

##--Define number of bins here--##
num_bins_lat = 8
num_bins_alt = 12

##--Determine CPC3 bin edges--##
CPC3_lat_bin_edges = np.linspace(CPC_df['Latitude'].min(), CPC_df['Latitude'].max(), num_bins_lat + 1)
CPC3_alt_bin_edges = np.linspace(CPC_df['Altitude'].min(), CPC_df['Altitude'].max(), num_bins_alt + 1)

##--Make 2d histogram for CPC3 and compute median particle abundance in each bin--##
CPC3_bin_medians, CPC3_x_edges, CPC3_y_edges, _ = binned_statistic_2d(CPC_df['Latitude'], 
    CPC_df['Altitude'], CPC_df['CPC3_conc'], statistic='median', bins=[CPC3_lat_bin_edges, CPC3_alt_bin_edges])

##--Histogram for CPC10--##
CPC10_lat_bin_edges = np.linspace(CPC_df['Latitude'].min(), CPC_df['Latitude'].max(), num_bins_lat + 1)
CPC10_alt_bin_edges = np.linspace(CPC_df['Altitude'].min(), CPC_df['Altitude'].max(), num_bins_alt +1)
CPC10_bin_medians, CPC10_x_edges, CPC10_y_edges, _ = binned_statistic_2d(CPC_df['Latitude'], 
    CPC_df['Altitude'], CPC_df['CPC10_conc'], statistic='median', bins=[CPC10_lat_bin_edges, CPC10_alt_bin_edges])

##--Histogram for nucleation mode particles--##
nuc_lat_bin_edges = np.linspace(CPC_df['Latitude'].min(), CPC_df['Latitude'].max(), num_bins_lat +1)
nuc_alt_bin_edges = np.linspace(CPC_df['Altitude'].min(), CPC_df['Altitude'].max(), num_bins_alt +1)
nuc_bin_medians, nuc_x_edges, nuc_y_edges, _ = binned_statistic_2d(CPC_df['Latitude'], 
    CPC_df['Altitude'], CPC_df['nuc_particles'], statistic='median', bins=[nuc_lat_bin_edges, nuc_alt_bin_edges])

################
##--PLOTTING--##
################

##--Particles larger than 3 nm--##
fig1, ax1 = plt.subplots(figsize=(8, 6))

##--Make special color map where 0 values are white--##
new_cmap = plt.get_cmap('viridis')
##--Values under specified minimum will be white--##
new_cmap.set_under('w')

##--Use pcolormesh for the plot, set minimum value for viridis colors as 1--##
CPC3_plot = ax1.pcolormesh(CPC3_x_edges, CPC3_y_edges, CPC3_bin_medians.T,  # Transpose to align correctly
    shading='auto', cmap=new_cmap, vmin=0, vmax=3500)

##--Add colorbar--##
cb = fig1.colorbar(CPC3_plot, ax=ax1)
cb.minorticks_on()
cb.set_label('Particles >2.5 nm $(Counts/cm^{3})$', fontsize=12)

# Set axis labels
ax1.set_xlabel('Latitude (°)', fontsize=12)
ax1.set_ylabel('Altitude (m)', fontsize=12)
ax1.set_title(f"Particles >2.5 nm Abundance - {flight.replace('Flight', 'Flight ')}")
#ax1.set_ylim(0, 6250)
#ax1.set_xlim(79.5, 83.7)

##--Base output path in directory--##
output_path = r"C:\Users\repooley\REP_PhD\NETCARE2015\data\processed\CurtainPlots\CPC3\AltitudeLatitude"

##--Use f-string to save file with flight# appended--##
output_path = f"{output_path}\\{flight}"
plt.savefig(output_path, dpi=600, bbox_inches='tight') 

plt.tight_layout()
plt.show()

##--Particles larger than 10 nm--##
fig2, ax2 = plt.subplots(figsize=(8, 6))

##--Use pcolormesh for the plot, set minimum for viridis colors as 1--##
CPC10_plot = ax2.pcolormesh(CPC10_x_edges, CPC10_y_edges, CPC10_bin_medians.T,  # Transpose to align correctly
    shading='auto', cmap=new_cmap, vmin=0, vmax=3000)

##--Add colorbar--##
cb2 = fig2.colorbar(CPC10_plot, ax=ax2)
cb2.minorticks_on()
cb2.set_label('Particles >10 nm $(Counts/cm^{3})$', fontsize=12)

##--Set axis labels--##
ax2.set_xlabel('Latitude (°)', fontsize=12)
ax2.set_ylabel('Altitude (m)', fontsize=12)
ax2.set_title(f"Particles >10 nm Abundance - {flight.replace('Flight', 'Flight ')}")
#ax2.set_ylim(0, 6250)
#ax2.set_xlim(79.5, 83.7)

##--Base output path in directory--##
output_path = r"C:\Users\repooley\REP_PhD\NETCARE2015\data\processed\CurtainPlots\CPC10\AltitudeLatitude"

##--Use f-string to save file with flight# appended--##
output_path = f"{output_path}\\{flight}"
plt.savefig(output_path, dpi=600, bbox_inches='tight') 

plt.tight_layout()
plt.show()

##--Nucleating particles--##
fig3, ax3 = plt.subplots(figsize=(8, 6))

##--Use pcolormesh for the plot and use viridis for values greater than 1--##
nuc_plot = ax3.pcolormesh(nuc_x_edges, nuc_y_edges, nuc_bin_medians.T,  # Transpose to align correctly
    shading='auto', cmap=new_cmap, vmin=0, vmax=1200)

##--Add colorbar--##
cb3 = fig3.colorbar(nuc_plot, ax=ax3)
cb3.minorticks_on()
cb.ax.tick_params(labelsize=16)
cb3.set_label('2.5-10 nm Particles $(Counts/cm^{3})$', fontsize=16)

##--Set axis labels--##
ax3.set_xlabel('Latitude (°)', fontsize=16)
ax3.set_ylabel('Altitude (m)', fontsize=16)
ax3.tick_params(axis='both', labelsize=16)
ax3.set_title(f"2.5-10 nm Particle Abundance - {flight.replace('Flight', 'Flight ')}", fontsize=18)
#ax3.set_ylim(0, 6250)
#ax3.set_xlim(79.5, 83.7)

##--Base output path in directory--##
output_path = r"C:\Users\repooley\REP_PhD\NETCARE2015\data\processed\CurtainPlots\Nucleating\AltitudeLatitude"

##--Use f-string to save file with flight# appended--##
output_path = f"{output_path}\\{flight}"
plt.savefig(output_path, dpi=600, bbox_inches='tight') 

plt.tight_layout()
plt.show()

########################
##--Diagnostic Plots--##
########################

##--Remove hashtags below to comment out this section--##
#'''

##--Counts per bin for CPC3 data--##
CPC3_bin_counts, _, _, _ = binned_statistic_2d(CPC_df['Latitude'], 
    CPC_df['Altitude'], CPC_df['CPC3_conc'], statistic='count', bins=[CPC3_lat_bin_edges, CPC3_alt_bin_edges])
 
##--Counts per bin for CPC10 data--##
CPC10_bin_counts, _, _, _ = binned_statistic_2d(CPC_df['Latitude'], 
    CPC_df['Altitude'], CPC_df['CPC10_conc'], statistic='count', bins=[CPC10_lat_bin_edges, CPC10_alt_bin_edges])
 
##--Counts per bin for N3-10 particles--##
nuc_bin_counts, _, _, _ = binned_statistic_2d(CPC_df['Latitude'], 
    CPC_df['Altitude'], CPC_df['nuc_particles'], statistic='count', bins=[nuc_lat_bin_edges, nuc_alt_bin_edges])

##--Plotting--##

##--Particles larger than 3 nm--##
fig1, ax1 = plt.subplots(figsize=(8, 6))

##--Make special color map where 0 values are white--##
new_cmap = plt.get_cmap('inferno')
##--Values under specified minimum will be white--##
new_cmap.set_under('w')

##--Use pcolormesh for the plot, set minimum value for viridis colors as 1--##
CPC3_diag_plot = ax1.pcolormesh(CPC3_x_edges, CPC3_y_edges, CPC3_bin_counts.T,  # Transpose to align correctly
    shading='auto', cmap=new_cmap, vmin=1, vmax=2000)

##--Add colorbar--##
cb = fig1.colorbar(CPC3_diag_plot, ax=ax1)
cb.minorticks_on()
cb.set_label('Number of Data Points', fontsize=12)

# Set axis labels
ax1.set_xlabel('Latitude (°)', fontsize=12)
ax1.set_ylabel('Altitude (m)', fontsize=12)
ax1.set_title(f"Particles >2.5 nm Counts per Bin - {flight.replace('Flight', 'Flight ')}")
#ax1.set_ylim(0, 6250)
#ax1.set_xlim(79.5, 83.7)

##--Base output path in directory--##
output_path = r"C:\Users\repooley\REP_PhD\NETCARE2015\data\processed\CurtainPlots\CPC3\AltitudeLatitude"

##--Use f-string to save file with flight# appended--##
output_path = f"{output_path}\\{flight}_diagnostic"
plt.savefig(output_path, dpi=600, bbox_inches='tight') 

plt.tight_layout()
plt.show()

##--Particles larger than 10 nm--##
fig2, ax2 = plt.subplots(figsize=(8, 6))

##--Use pcolormesh for the plot, set minimum for viridis colors as 1--##
CPC10_diag_plot = ax2.pcolormesh(CPC10_x_edges, CPC10_y_edges, CPC10_bin_counts.T,  # Transpose to align correctly
    shading='auto', cmap=new_cmap, vmin=1, vmax=2000)

##--Add colorbar--##
cb2 = fig2.colorbar(CPC10_diag_plot, ax=ax2)
cb2.minorticks_on()
cb2.set_label('Number of Data Points', fontsize=12)

##--Set axis labels--##
ax2.set_xlabel('Latitude (°)', fontsize=12)
ax2.set_ylabel('Altitude (m)', fontsize=12)
ax2.set_title(f"Particles >10 nm Counts per Bin - {flight.replace('Flight', 'Flight ')}")
#ax2.set_ylim(0, 6250)
#ax2.set_xlim(79.5, 83.7)

##--Base output path in directory--##
output_path = r"C:\Users\repooley\REP_PhD\NETCARE2015\data\processed\CurtainPlots\CPC10\AltitudeLatitude"

##--Use f-string to save file with flight# appended--##
output_path = f"{output_path}\\{flight}_diagnostic"
plt.savefig(output_path, dpi=600, bbox_inches='tight') 

plt.tight_layout()
plt.show()

##--Nucleating particles--##
fig3, ax3 = plt.subplots(figsize=(8, 6))

##--Use pcolormesh for the plot and use viridis for values greater than 1--##
nuc_diag_plot = ax3.pcolormesh(nuc_x_edges, nuc_y_edges, nuc_bin_counts.T,  # Transpose to align correctly
    shading='auto', cmap=new_cmap, vmin=1, vmax=2000)

##--Add colorbar--##
cb3 = fig3.colorbar(nuc_diag_plot, ax=ax3)
cb3.minorticks_on()
cb3.set_label('Number of Data Points', fontsize=12)

##--Set axis labels--##
ax3.set_xlabel('Latitude (°)', fontsize=12)
ax3.set_ylabel('Altitude (m)', fontsize=12)
ax3.set_title(f"2.5-10 nm Particle Counts per Bin - {flight.replace('Flight', 'Flight ')}")
#ax3.set_ylim(0, 6250)
#ax3.set_xlim(79.5, 83.7)

##--Base output path in directory--##
output_path = r"C:\Users\repooley\REP_PhD\NETCARE2015\data\processed\CurtainPlots\Nucleating\AltitudeLatitude"

##--Use f-string to save file with flight# appended--##
output_path = f"{output_path}\\{flight}_diagnostic"
plt.savefig(output_path, dpi=600, bbox_inches='tight') 

plt.tight_layout()
plt.show()
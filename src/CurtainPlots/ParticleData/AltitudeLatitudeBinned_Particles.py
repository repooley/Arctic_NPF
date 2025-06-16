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

###################
##--User inputs--##
###################

##--Set the base directory to project folder--##
directory = r"C:\Users\repooley\REP_PhD\NETCARE2015\data"
 
##--Select flight (Flight1 thru Flight10)--##
flight = "Flight10" 

##--Set binning for PTemp and Latitude--##
num_bins_lat = 4
num_bins_alt = 8

##--Bin data are in a CSV file--##
UHSAS_bins = pd.read_csv(r"C:\Users\repooley\REP_PhD\NETCARE2015\data\raw\NETCARE2015_UHSAS_bins.csv")

##--Base output path for figures in directory--##
output_path = r"C:\Users\repooley\REP_PhD\NETCARE2015\data\processed\CurtainPlots"

#########################
##--Open ICARTT Files--##
#########################

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

##--Not pulling properly for flight 6, unsure why--##
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

######################
##--Calc N(2.5-10)--##
######################

##--Convert to STP--##
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

##--Creates a Pandas dataframe for CPC data--##
CPC_df = pd.DataFrame({'CPC3_conc':CPC3_conc_STP, 'CPC10_conc': CPC10_conc_STP})

##--Calculate N3-10 particles--##
nuc_particles = (CPC_df['CPC3_conc'] - CPC_df['CPC10_conc'])

##--Change calculated particle counts less than zero to NaN--##
nuc_particles = np.where(nuc_particles >= 0, nuc_particles, np.nan)

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

###########################
##--Create 2D histogram--##
###########################

##--Float type NaNs in potential_temp cannot convert to int, so must be removed--##
CPC3_df = pd.DataFrame({'Altitude': altitude, 'Latitude': latitude, 'CPC3':CPC3_conc_STP})
CPC3_clean_df = CPC3_df.dropna()

##--Make separate dfs to preserve as much data as possible--##
CPC10_df = pd.DataFrame({'Altitude': altitude, 'Latitude': latitude, 'CPC10': CPC10_conc_STP})
CPC10_clean_df = CPC10_df.dropna()

##--2.5-10 nm, nucleating--##
nuc_df = pd.DataFrame({'Altitude': altitude, 'Latitude': latitude, 'nuc_particles': nuc_particles})
nuc_clean_df = nuc_df.dropna()

##--10-89, growth--##
grow_df = pd.DataFrame({'Altitude': altitude, 'Latitude': latitude, 'n_10_89': n_10_89})
grow_clean_df = grow_df.dropna()

##--Compute global min/max values across all data BEFORE dropping NaNs--##
lat_min, lat_max = np.nanmin(latitude), np.nanmax(latitude)
alt_min, alt_max = np.nanmin(altitude), np.nanmax(altitude)

##--Generate common bin edges using specified number of bins--##
common_lat_bin_edges = np.linspace(lat_min, lat_max, num_bins_lat + 1)
common_alt_bin_edges = np.linspace(alt_min, alt_max, num_bins_alt + 1)

##--Make 2D histograms using common bins--##
##--CPC3--##
CPC3_bin_medians, _, _, _ = binned_statistic_2d(CPC3_clean_df['Latitude'], 
    CPC3_clean_df['Altitude'], CPC3_clean_df['CPC3'], statistic='median', bins=[common_lat_bin_edges, common_alt_bin_edges])

##--CPC10--##
CPC10_bin_medians, _, _, _ = binned_statistic_2d(CPC10_clean_df['Latitude'], 
    CPC10_clean_df['Altitude'], CPC10_clean_df['CPC10'], statistic='median', bins=[common_lat_bin_edges, common_alt_bin_edges])

##--N(2.5-10)--##
nuc_bin_medians, _, _, _ = binned_statistic_2d(nuc_clean_df['Latitude'], 
    nuc_clean_df['Altitude'], nuc_clean_df['nuc_particles'], statistic='median', bins=[common_lat_bin_edges, common_alt_bin_edges])

##--N(10-89)--##
grow_bin_medians, _, _, _ = binned_statistic_2d(grow_clean_df['Latitude'], 
    grow_clean_df['Altitude'], grow_clean_df['n_10_89'], statistic='median', bins=[common_lat_bin_edges, common_alt_bin_edges])

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
CPC3_plot = ax1.pcolormesh(common_lat_bin_edges, common_alt_bin_edges, CPC3_bin_medians.T,  # Transpose to align correctly
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

##--Use f-string to save file with flight# appended--##
CPC3_output_path = f"{output_path}\\CPC3/AltitudeLatitude/{flight}"
plt.savefig(CPC3_output_path, dpi=600, bbox_inches='tight') 

plt.tight_layout()
plt.show()

##--Particles larger than 10 nm--##
fig2, ax2 = plt.subplots(figsize=(8, 6))

##--Use pcolormesh for the plot, set minimum for viridis colors as 1--##
CPC10_plot = ax2.pcolormesh(common_lat_bin_edges, common_alt_bin_edges, CPC10_bin_medians.T,  # Transpose to align correctly
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

##--Use f-string to save file with flight# appended--##
CPC10_output_path = f"{output_path}\\CPC10/AltitudeLatitude/{flight}"
plt.savefig(CPC10_output_path, dpi=600, bbox_inches='tight') 

plt.tight_layout()
plt.show()

##--Nucleating particles--##
fig3, ax3 = plt.subplots(figsize=(8, 6))

##--Use pcolormesh for the plot and use viridis for values greater than 1--##
nuc_plot = ax3.pcolormesh(common_lat_bin_edges, common_alt_bin_edges, nuc_bin_medians.T,  # Transpose to align correctly
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

##--Use f-string to save file with flight# appended--##
nuc_output_path = f"{output_path}\\Nucleating/AltitudeLatitude/{flight}"
plt.savefig(nuc_output_path, dpi=600, bbox_inches='tight') 

plt.tight_layout()
plt.show()

##--10-89nm particles--##
fig4, ax4 = plt.subplots(figsize=(8, 6))

##--Use pcolormesh for the plot and use viridis for values greater than 1--##
grow_plot = ax4.pcolormesh(common_lat_bin_edges, common_alt_bin_edges, grow_bin_medians.T,  # Transpose to align correctly
    shading='auto', cmap=new_cmap, vmin=0, vmax=1200)

##--Add colorbar--##
cb4 = fig4.colorbar(grow_plot, ax=ax4)
cb4.minorticks_on()
cb.ax.tick_params(labelsize=16)
cb4.set_label('10-89 nm Particles $(Counts/cm^{3})$', fontsize=16)

##--Set axis labels--##
ax4.set_xlabel('Latitude (°)', fontsize=16)
ax4.set_ylabel('Altitude (m)', fontsize=16)
ax4.tick_params(axis='both', labelsize=16)
ax4.set_title(f"10-89 nm Particle Abundance - {flight.replace('Flight', 'Flight ')}", fontsize=18)
#ax4.set_ylim(0, 6250)
#ax4.set_xlim(79.5, 83.7)

##--Use f-string to save file with flight# appended--##
nuc_output_path = f"{output_path}\\N_10_89/AltitudeLatitude/{flight}"
plt.savefig(nuc_output_path, dpi=600, bbox_inches='tight') 

plt.tight_layout()
plt.show()

########################
##--Diagnostic Plots--##
########################

##--Remove hashtags below to comment out this section--##
#'''

##--Counts per bin for CPC3 data--##
CPC3_bin_counts, _, _, _ = binned_statistic_2d(CPC_df['Latitude'], 
    CPC_df['Altitude'], CPC_df['CPC3_conc'], statistic='count', bins=[common_lat_bin_edges, common_alt_bin_edges])
 
##--Counts per bin for CPC10 data--##
CPC10_bin_counts, _, _, _ = binned_statistic_2d(CPC_df['Latitude'], 
    CPC_df['Altitude'], CPC_df['CPC10_conc'], statistic='count', bins=[common_lat_bin_edges, common_alt_bin_edges])
 
##--Counts per bin for N3-10 particles--##
nuc_bin_counts, _, _, _ = binned_statistic_2d(CPC_df['Altitude'], 
    CPC_df['nuc_particles'], statistic='count', bins=[common_lat_bin_edges, common_alt_bin_edges])

##--Counts per bin for N10-89 particles--##
grow_bin_counts, _, _, _ = binned_statistic_2d(CPC_df['Latitude'], 
    CPC_df['Altitude'], CPC_df['n_10_89'], statistic='count', bins=[common_lat_bin_edges, common_alt_bin_edges])

##--Plotting--##

##--Particles larger than 3 nm--##
fig1, ax1 = plt.subplots(figsize=(8, 6))

##--Make special color map where 0 values are white--##
new_cmap = plt.get_cmap('inferno')
##--Values under specified minimum will be white--##
new_cmap.set_under('w')

##--Use pcolormesh for the plot, set minimum value for viridis colors as 1--##
CPC3_diag_plot = ax1.pcolormesh(common_lat_bin_edges, common_alt_bin_edges, CPC3_bin_counts.T,  # Transpose to align correctly
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

##--Use f-string to save file with flight# appended--##
CPC3_diag_output_path = f"{output_path}\\CPC3/AltitudeLatitude/{flight}_diagnostic"
plt.savefig(CPC3_diag_output_path, dpi=600, bbox_inches='tight') 

plt.tight_layout()
plt.show()

##--Particles larger than 10 nm--##
fig2, ax2 = plt.subplots(figsize=(8, 6))

##--Use pcolormesh for the plot, set minimum for viridis colors as 1--##
CPC10_diag_plot = ax2.pcolormesh(common_lat_bin_edges, common_alt_bin_edges, CPC10_bin_counts.T,  # Transpose to align correctly
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

##--Use f-string to save file with flight# appended--##
CPC10_diag_output_path = f"{output_path}\\CPC10/AltitudeLatitude/{flight}_diagnostic"
plt.savefig(CPC10_diag_output_path, dpi=600, bbox_inches='tight') 

plt.tight_layout()
plt.show()

##--Nucleating particles--##
fig3, ax3 = plt.subplots(figsize=(8, 6))

##--Use pcolormesh for the plot and use viridis for values greater than 1--##
nuc_diag_plot = ax3.pcolormesh(common_lat_bin_edges, common_alt_bin_edges, nuc_bin_counts.T,  # Transpose to align correctly
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

##--Use f-string to save file with flight# appended--##
nuc_diag_output_path = f"{output_path}\\Nucleating/AltitudeLatitude/{flight}_diagnostic"
plt.savefig(nuc_diag_output_path, dpi=600, bbox_inches='tight') 

plt.tight_layout()
plt.show()

##--10-85nm particles--##
fig3, ax3 = plt.subplots(figsize=(8, 6))

##--Use pcolormesh for the plot and use viridis for values greater than 1--##
grow_diag_plot = ax3.pcolormesh(common_lat_bin_edges, common_alt_bin_edges, grow_bin_counts.T,  # Transpose to align correctly
    shading='auto', cmap=new_cmap, vmin=1, vmax=2000)

##--Add colorbar--##
cb3 = fig3.colorbar(nuc_diag_plot, ax=ax3)
cb3.minorticks_on()
cb3.set_label('Number of Data Points', fontsize=12)

##--Set axis labels--##
ax3.set_xlabel('Latitude (°)', fontsize=12)
ax3.set_ylabel('Altitude (m)', fontsize=12)
ax3.set_title(f"10-89 nm Particle Counts per Bin - {flight.replace('Flight', 'Flight ')}")
#ax3.set_ylim(0, 6250)
#ax3.set_xlim(79.5, 83.7)

##--Use f-string to save file with flight# appended--##
nuc_diag_output_path = f"{output_path}\\N_10_89/AltitudeLatitude/{flight}_diagnostic"
plt.savefig(nuc_diag_output_path, dpi=600, bbox_inches='tight') 

plt.tight_layout()
plt.show()
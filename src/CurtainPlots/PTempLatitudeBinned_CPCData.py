# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 10:56:07 2025

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

##--Select flight to analyze (Flight1 thru Flight10)--##
flight = "Flight1"

##--Set binning for PTemp and Latitude--##
num_bins_lat = 4
num_bins_ptemp = 12

##--Base output path for figures in directory--##
output_path = r"C:\Users\repooley\REP_PhD\NETCARE2015\data\processed\CurtainPlots\Nucleating\PTempLatitude"

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
nuc_clean_df = nuc_df.dropna()

##--Compute global min/max values across all data BEFORE dropping NaNs--##
lat_min, lat_max = np.nanmin(latitude), np.nanmax(latitude)
ptemp_min, ptemp_max = np.nanmin(potential_temp), np.nanmax(potential_temp)

##--Generate common bin edges using specified number of bins--##
common_lat_bin_edges = np.linspace(lat_min, lat_max, num_bins_lat + 1)
common_ptemp_bin_edges = np.linspace(ptemp_min, ptemp_max, num_bins_ptemp + 1)

##--Make 2D histograms using common bins--##
##--CPC3--##
CPC3_bin_medians, _, _, _ = binned_statistic_2d(CPC3_clean_df['Latitude'], 
    CPC3_clean_df['PTemp'], CPC3_clean_df['CPC3'], statistic='median', bins=[common_lat_bin_edges, common_ptemp_bin_edges])

##--CPC10--##
CPC10_bin_medians, _, _, _ = binned_statistic_2d(CPC10_clean_df['Latitude'], 
    CPC10_clean_df['PTemp'], CPC10_clean_df['CPC10'], statistic='median', bins=[common_lat_bin_edges, common_ptemp_bin_edges])

##--N(2.5-10)--##
nuc_bin_medians, _, _, _ = binned_statistic_2d(nuc_clean_df['Latitude'], 
    nuc_clean_df['PTemp'], nuc_clean_df['nuc_particles'], statistic='median', bins=[common_lat_bin_edges, common_ptemp_bin_edges])

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
CPC3_plot = ax1.pcolormesh(common_lat_bin_edges, common_ptemp_bin_edges, CPC3_bin_medians.T,  # Transpose to align correctly
    shading='auto', cmap=new_cmap, vmin=0, vmax=2500)

##--Add dashed horizontal lines for the polar dome boundaries--##
ax1.axhline(y=275, color='k', linestyle='--', linewidth=1)
ax1.axhline(y=299, color='k', linestyle='--', linewidth=1)

##--Add colorbar--##
cb = fig1.colorbar(CPC3_plot, ax=ax1)
cb.minorticks_on()
cb.ax.tick_params(labelsize=16)
cb.set_label('Particles >2.5 nm $(Counts/cm^{3})$', fontsize=16)

##--Set axis labels--##
ax1.set_xlabel('Latitude (°)', fontsize=16)
ax1.set_ylabel('Potential Temperature \u0398 (K)', fontsize=16)
ax1.tick_params(axis='both', labelsize=16)
ax1.set_title(f"Particles >2.5 nm Abundance - {flight.replace('Flight', 'Flight ')}", fontsize=18)
#ax1.set_ylim(238, 301)
#ax1.set_xlim(79.5, 83.7)

##--Use f-string to save file with flight# appended--##
CPC3_output_path = f"{output_path}\\{flight}"
plt.savefig(CPC3_output_path, dpi=600, bbox_inches='tight') 

plt.tight_layout()
plt.show()

##--Particles larger than 10 nm--##
fig2, ax2 = plt.subplots(figsize=(8, 6))

##--Use pcolormesh for the plot, set minimum for viridis colors as 1--##
CPC10_plot = ax2.pcolormesh(common_lat_bin_edges, common_ptemp_bin_edges, CPC10_bin_medians.T,  # Transpose to align correctly
    shading='auto', cmap=new_cmap, vmin=0, vmax=2000)

##--Add dashed horizontal lines for the polar dome boundaries--##
ax2.axhline(y=275, color='k', linestyle='--', linewidth=1)
ax2.axhline(y=299, color='k', linestyle='--', linewidth=1)

##--Add colorbar--##
cb2 = fig2.colorbar(CPC10_plot, ax=ax2)
cb2.minorticks_on()
cb2.ax.tick_params(labelsize=16)
cb2.set_label('Particles >10 nm $(Counts/cm^{3})$', fontsize=16)

##--Set axis labels--##
ax2.set_xlabel('Latitude (°)', fontsize=16)
ax2.set_ylabel('Potential Temperature \u0398 (K)', fontsize=16)
ax2.tick_params(axis='both', labelsize=16)
ax2.set_title(f"Particles >10 nm Abundance - {flight.replace('Flight', 'Flight ')}", fontsize=18)
#ax2.set_ylim(238, 301)
#ax2.set_xlim(79.5, 83.7)

##--Use f-string to save file with flight# appended--##
CPC10_output_path = f"{output_path}\\{flight}"
plt.savefig(CPC10_output_path, dpi=600, bbox_inches='tight') 

plt.tight_layout()
plt.show()

##--Nucleating particles--##
fig3, ax3 = plt.subplots(figsize=(8, 6))

##--Use pcolormesh for the plot and use viridis for values greater than 1--##
nuc_plot = ax3.pcolormesh(common_lat_bin_edges, common_ptemp_bin_edges, nuc_bin_medians.T,  # Transpose to align correctly
    shading='auto', cmap=new_cmap, vmin=0, vmax=1000)

##--Add dashed horizontal lines for the polar dome boundaries--##
ax3.axhline(y=275, color='k', linestyle='--', linewidth=1)
ax3.axhline(y=299, color='k', linestyle='--', linewidth=1)

##--Add colorbar--##
cb3 = fig3.colorbar(nuc_plot, ax=ax3)
cb3.minorticks_on()
cb3.ax.tick_params(labelsize=16)
cb3.set_label('2.5-10 nm Particles $(Counts/cm^{3})$', fontsize=16)

##--Set axis labels--##
ax3.set_xlabel('Latitude (°)', fontsize=16)
ax3.set_ylabel('Potential Temperature \u0398 (K)', fontsize=16)
ax3.tick_params(axis='both', labelsize=16)
ax3.set_title(f"2.5-10 nm Particle Abundance - {flight.replace('Flight', 'Flight ')}", fontsize=18)
#ax3.set_ylim(238, 301)
#ax3.set_xlim(79.5, 83.7)

##--Use f-string to save file with flight# appended--##
nuc_output_path = f"{output_path}\\{flight}"
plt.savefig(nuc_output_path, dpi=600, bbox_inches='tight') 

plt.tight_layout()
plt.show()

########################
##--Diagnostic Plots--##
########################

##--Remove hashtags below to comment out this section--##
#'''

##--Counts per bin for CPC3 data--##
CPC3_bin_counts, _, _, _ = binned_statistic_2d(CPC3_clean_df['Latitude'], 
    CPC3_clean_df['PTemp'], CPC3_clean_df['CPC3'], statistic='count', bins=[common_lat_bin_edges, common_ptemp_bin_edges])
 
##--Counts per bin for CPC10 data--##
CPC10_bin_counts, _, _, _ = binned_statistic_2d(CPC10_clean_df['Latitude'], 
    CPC10_clean_df['PTemp'], CPC10_clean_df['CPC10'], statistic='count', bins=[common_lat_bin_edges, common_ptemp_bin_edges])

##--Counts per bin for N3-10 particles--##
nuc_bin_counts, _, _, _ = binned_statistic_2d(nuc_clean_df['Latitude'], 
    nuc_clean_df['PTemp'], nuc_clean_df['nuc_particles'], statistic='count', bins=[common_lat_bin_edges, common_ptemp_bin_edges])

##--Plotting--##

##--Particles larger than 3 nm--##
fig1, ax1 = plt.subplots(figsize=(8, 6))

##--Make special color map where 0 values are white--##
new_cmap = plt.get_cmap('inferno')
##--Values under specified minimum will be white--##
new_cmap.set_under('w')

##--Use pcolormesh for the plot, set minimum value for viridis colors as 1--##
CPC3_plot = ax1.pcolormesh(common_lat_bin_edges, common_ptemp_bin_edges, CPC3_bin_counts.T,  # Transpose to align correctly
    shading='auto', cmap=new_cmap, vmin=1, vmax=1250)

##--Add colorbar--##
cb = fig1.colorbar(CPC3_plot, ax=ax1)
cb.minorticks_on()
cb.ax.tick_params(labelsize=16)
cb.set_label('Number of Data Points', fontsize=16)

# Set axis labels
ax1.set_xlabel('Latitude (°)', fontsize=16)
ax1.set_ylabel('Potential Temperature \u0398 (K)', fontsize=16)
ax1.tick_params(axis='both', labelsize=16)
ax1.set_title(f"Particles >2.5 nm Counts per Bin - {flight.replace('Flight', 'Flight ')}", fontsize=18)
#ax1.set_ylim(238, 301)
#ax1.set_xlim(79.5, 83.7)

##--Use f-string to save file with flight# appended--##
CPC3_diag_output_path = f"{output_path}\\{flight}_diagnostic"
plt.savefig(CPC3_diag_output_path, dpi=600, bbox_inches='tight') 

plt.tight_layout()
plt.show()

##--Particles larger than 10 nm--##
fig2, ax2 = plt.subplots(figsize=(8, 6))

##--Use pcolormesh for the plot, set minimum for viridis colors as 1--##
CPC10_plot = ax2.pcolormesh(common_lat_bin_edges, common_ptemp_bin_edges, CPC10_bin_counts.T,  # Transpose to align correctly
    shading='auto', cmap=new_cmap, vmin=1, vmax=1250)

##--Add colorbar--##
cb2 = fig2.colorbar(CPC10_plot, ax=ax2)
cb2.minorticks_on()
cb2.ax.tick_params(labelsize=16)
cb2.set_label('Number of Data Points', fontsize=16)

##--Set axis labels--##
ax2.set_xlabel('Latitude (°)', fontsize=16)
ax2.set_ylabel('Potential Temperature \u0398 (K)', fontsize=16)
ax2.tick_params(axis='both', labelsize=16)
ax2.set_title(f"Particles >10 nm Counts per Bin - {flight.replace('Flight', 'Flight ')}", fontsize=18)
#ax2.set_ylim(238, 301)
#ax2.set_xlim(79.5, 83.7)

##--Use f-string to save file with flight# appended--##
CPC10_diag_output_path = f"{output_path}\\{flight}_diagnostic"
plt.savefig(CPC10_diag_output_path, dpi=600, bbox_inches='tight') 

plt.tight_layout()
plt.show()

##--Nucleating particles--##
fig3, ax3 = plt.subplots(figsize=(8, 6))

##--Use pcolormesh for the plot and use viridis for values greater than 1--##
nuc_plot = ax3.pcolormesh(common_lat_bin_edges, common_ptemp_bin_edges, nuc_bin_counts.T,  # Transpose to align correctly
    shading='auto', cmap=new_cmap, vmin=1, vmax=1000)

##--Add colorbar--##
cb3 = fig3.colorbar(nuc_plot, ax=ax3)
cb3.minorticks_on()
cb3.ax.tick_params(labelsize=16)
cb3.set_label('Number of Data Points', fontsize=16)

##--Set axis labels--##
ax3.set_xlabel('Latitude (°)', fontsize=16)
ax3.set_ylabel('Potential Temperature \u0398 (K)', fontsize=16)
ax3.tick_params(axis='both', labelsize=16)
ax3.set_title(f"2.5-10 nm Particle Counts per Bin - {flight.replace('Flight', 'Flight ')}", fontsize=18)
#ax3.set_ylim(238, 301)
#ax3.set_xlim(79.5, 83.7)

##--Use f-string to save file with flight# appended--##
nuc_diag_output_path = f"{output_path}\\{flight}_diagnostic"
plt.savefig(nuc_diag_output_path, dpi=600, bbox_inches='tight') 

plt.tight_layout()
plt.show()
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 16:38:53 2025

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
flight = "Flight3" # Flight1 AIMMS file currently broken at line 13234

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

##--Water mixing data--##
H2O = icartt.Dataset(find_files(directory, flight, 'H2O')[0])

#################
##--Pull data--##
#################

##--AIMMS Data--##
altitude = aimms.data['Alt'] # in m
latitude = aimms.data['Lat'] # in degrees
temperature = aimms.data['Temp'] # in C
pressure = aimms.data['BP'] # in pa
aimms_time =aimms.data['TimeWave'] # seconds since midnight

##--10 nm CPC data--##
H2O_time = H2O.data['Time_UTC'] # seconds since midnight
H2O_conc = H2O.data['H2O_ppmv'] # ppmv

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

##--Trim H2O data if it starts before AIMMS--##
if H2O_time.min() < aimms_start:
    mask_start = H2O_time >= aimms_start
    H2O_time = H2O_time[mask_start]
    H2O_conc = H2O_conc[mask_start]
    
##--Append H2O data with NaNs if it ends before AIMMS--##
if H2O_time.max() < aimms_end: 
    missing_times = np.arange(H2O_time.max()+1, aimms_end +1)
    H2O_time = np.concatenate([H2O_time, missing_times])
    H2O_conc = np.concatenate([H2O_conc, [np.nan]*len(missing_times)])

##--Create a DataFrame for H2O data and reindex to AIMMS time, setting non-overlapping times to nan--##
H2O_df = pd.DataFrame({'Time_UTC': H2O_time, 'H2O_ppmv': H2O_conc})
H2O_aligned = H2O_df.set_index('Time_UTC').reindex(aimms_time)
H2O_aligned['H2O_ppmv']=H2O_aligned['H2O_ppmv'].where(H2O_aligned.index.isin(aimms_time), np.nan)
H2O_conc_aligned = H2O_aligned['H2O_ppmv']

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

###################
##--Conversions--##
###################

##--Convert H2O ppm to RH wrt Water--##

##--Lowe and Ficke (1974) 6th deg polynomial approach--##
##--Sat vap pressure water -50 to 50 C--##
wa0 = 6.107799961
wa1 = 4.436518521E-1
wa2 = 1.428945805E-2
wa3 = 2.650648471E-4
wa4 = 3.031240396E-6
wa5 = 2.034080948E-8
wa6 = 6.136820929E-11

##--Generate empty lists for RH wrt water outputs--##
saturation_humidity_w = []
relative_humidity_w = []

##--Calculate saturation humidity in ppmv and relative humidity--##
for T, P, H2O_ppmv in zip(temperature, pressure, H2O_conc_aligned):
    ##--Only calculate within temp range--##
    if -50 <= T < 50:
        ##--saturation vapor pressure using Lowe and Ficke (1974) eqn--##
        e_sw = wa0 + wa1*T + wa2*(T**2)+ wa3*(T**3)+ wa4*(T**4) + wa5*(T**5) + wa6*(T**6) # in mbar 
        ##--Convert from mbar to pa--##
        e_sw_pa = e_sw*100
        ##--Saturation mixing ratio in ppmv--##
        w_s_ppmv = (e_sw_pa / P) * 1e6
        saturation_humidity_w.append(w_s_ppmv)
        ##--Relative humidity--##
        RH = (H2O_ppmv / w_s_ppmv) * 100  # in %
        relative_humidity_w.append(RH)
    else:
        saturation_humidity_w.append(np.nan)  
        relative_humidity_w.append(np.nan)    

##--With respect to ice--##

##--Lowe and Ficke (1974) 6th deg polynomial approach--##
##--Sat vap pressure ice -50 to 0 C--##
ia0 = 6.109177956
ia1 = 5.034698970E-1
ia2 = 1.886013408E-2
ia3 = 4.176223716E-4
ia4 = 5.824720280E-6
ia5 = 4.838803174E-8
ia6 = 1.838826904E-10

##--Generate empty lists for humidity outputs--##
saturation_humidity_i = []
relative_humidity_i = []

##--Calculate saturation humidity wrt ice in ppmv and RH--##
for T, P, H2O_ppmv in zip(temperature, pressure, H2O_conc_aligned):
    ##--Only calculate within temp range--##
    if -50 <= T < 0:
        ##--Saturation vapor pressure using Lowe and Ficke (1974) eqn--##
        e_si = ia0 + ia1*T + ia2*(T**2) + ia3*(T**3) + ia4*(T**4) + ia5*(T**5) + ia6*(T**6)  # in mbar
        ##--Convert from mbar to Pa--##
        e_si_pa = e_si * 100
        ##--Saturation mixing ratio in ppbv--##
        e_si_ppmv = (e_si_pa / P) * 1e6
        saturation_humidity_i.append(e_si_ppmv)
        ##--Relative Humidity--##
        RH_i = (H2O_ppmv / e_si_ppmv) * 100  # in %
        relative_humidity_i.append(RH_i)
    else:
        saturation_humidity_i.append(np.nan)  
        relative_humidity_i.append(np.nan)    

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

##--Convert particle data to STP--##

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

##--Creates separate dfs to preserve data--##
##--Including nuc_particles downsizes dataset to instances of N3-10. Comment out if full dataset desired--##
w_df = pd.DataFrame({'PTemp': potential_temp, 'Latitude': latitude, 'Relative_Humidity_w': relative_humidity_w})#, 'nuc_particles': nuc_particles})
i_df = pd.DataFrame({'PTemp': potential_temp, 'Latitude': latitude, 'Relative_Humidity_i': relative_humidity_i})#, 'nuc_particles': nuc_particles})
temp_df = pd.DataFrame({'PTemp': potential_temp, 'Latitude': latitude, 'Temperature': temperature_k})#, 'nuc_particles': nuc_particles})

##--Drop NaNs to prevent issues with potential_temp floats--##
clean_w_df = w_df.dropna()
clean_i_df = i_df.dropna()
clean_temp_df = temp_df.dropna()

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
RH_w_bin_medians, _, _, _ = binned_statistic_2d(
    clean_w_df['Latitude'], clean_w_df['PTemp'], clean_w_df['Relative_Humidity_w'], 
    statistic='median', bins=[common_lat_bin_edges, common_ptemp_bin_edges])

RH_i_bin_medians, _, _, _ = binned_statistic_2d(
    clean_i_df['Latitude'], clean_i_df['PTemp'], clean_i_df['Relative_Humidity_i'], 
    statistic='median', bins=[common_lat_bin_edges, common_ptemp_bin_edges])

temp_bin_medians, _, _, _ = binned_statistic_2d(
    clean_temp_df['Latitude'], clean_temp_df['PTemp'], clean_temp_df['Temperature'], 
    statistic='median', bins=[common_lat_bin_edges, common_ptemp_bin_edges])

################
##--PLOTTING--##
################

##--RH wrt water--##
fig1, ax1 = plt.subplots(figsize=(8, 6))

##--Make special color map where 0 values are white--##
new_cmap = plt.get_cmap('viridis')
##--Values under specified minimum will be white--##
new_cmap.set_under('w')

RH_w_plot = ax1.pcolormesh(common_lat_bin_edges, common_ptemp_bin_edges, RH_w_bin_medians.T,  
    shading='auto', cmap=new_cmap, vmin=0, vmax=120)

##--Add dashed horizontal lines for the polar dome boundaries--##
ax1.axhline(y=275, color='k', linestyle='--', linewidth=1)
ax1.axhline(y=299, color='k', linestyle='--', linewidth=1)

##--Add colorbar--##
cb = fig1.colorbar(RH_w_plot, ax=ax1)
cb.minorticks_on()
cb.ax.tick_params(labelsize=16)
cb.set_label('Relative Humidity', fontsize=16)

# Set axis labels
ax1.set_xlabel('Latitude (°)', fontsize=16)
ax1.set_ylabel('Potential Temperature \u0398 (K)', fontsize=16)
ax1.tick_params(axis='both', labelsize=16)
ax1.set_title(f"Relative Humidity wrt Water - {flight.replace('Flight', 'Flight ')}", fontsize=18)
ax1.set_ylim(244, 301)
#ax1.set_xlim(82.4, 83.4)

##--Base output path in directory--##
output_path = r"C:\Users\repooley\REP_PhD\NETCARE2015\data\processed\CurtainPlots\Meteorological\PTempLatitude"

##--Use f-string to save file with flight# appended--##
output_path = f"{output_path}\\{flight}_RHwrtWater.png"
plt.savefig(output_path, dpi=600, bbox_inches='tight') 
plt.tight_layout()
plt.show()

##--RH wrt ice--##
fig2, ax2 = plt.subplots(figsize=(8, 6))

##--Use pcolormesh for the plot, set minimum value for viridis colors as 1--##
RH_i_plot = ax2.pcolormesh(common_lat_bin_edges, common_ptemp_bin_edges, RH_i_bin_medians.T,  # Transpose to align correctly
    shading='auto', cmap=new_cmap, vmin=0, vmax=120)

##--Add dashed horizontal lines for the polar dome boundaries--##
ax2.axhline(y=275, color='k', linestyle='--', linewidth=1)
ax2.axhline(y=299, color='k', linestyle='--', linewidth=1)

##--Add colorbar--##
cb = fig2.colorbar(RH_i_plot, ax=ax2)
cb.minorticks_on()
cb.ax.tick_params(labelsize=16)
cb.set_label('Relative Humidity', fontsize=16)

# Set axis labels
ax2.set_xlabel('Latitude (°)', fontsize=16)
ax2.set_ylabel('Potential Temperature \u0398 (K)', fontsize=16)
ax2.tick_params(axis='both', labelsize=16)
ax2.set_title(f"Relative Humidity wrt Ice - {flight.replace('Flight', 'Flight ')}", fontsize=18)
ax2.set_ylim(244, 301)
#ax2.set_xlim(82.4, 83.4)

##--Base output path in directory--##
output_path = r"C:\Users\repooley\REP_PhD\NETCARE2015\data\processed\CurtainPlots\Meteorological\PTempLatitude"

##--Use f-string to save file with flight# appended--##
output_path = f"{output_path}\\{flight}_RHwrtIce.png"
plt.savefig(output_path, dpi=600, bbox_inches='tight') 
plt.tight_layout()
plt.show()

##--Temperature--##
fig3, ax3 = plt.subplots(figsize=(8, 6))

##--Use pcolormesh for the plot, set minimum value for viridis colors as 1--##
temp_plot = ax3.pcolormesh(common_lat_bin_edges, common_ptemp_bin_edges, temp_bin_medians.T,  # Transpose to align correctly
    shading='auto', cmap=new_cmap, vmin=200, vmax=300)

##--Add dashed horizontal lines for the polar dome boundaries--##
ax3.axhline(y=275, color='k', linestyle='--', linewidth=1)
ax3.axhline(y=299, color='k', linestyle='--', linewidth=1)

##--Add colorbar--##
cb = fig3.colorbar(temp_plot, ax=ax3)
cb.minorticks_on()
cb.ax.tick_params(labelsize=16)
cb.set_label('Temperature (K)', fontsize=16)

# Set axis labels
ax3.set_xlabel('Latitude (°)', fontsize=16)
ax3.set_ylabel('Potential Temperature \u0398 (K)', fontsize=16)
ax3.tick_params(axis='both', labelsize=16)
ax3.set_title(f"Absolute Temperature - {flight.replace('Flight', 'Flight ')}", fontsize=18)
ax3.set_ylim(244, 301)
#ax3.set_xlim(82.4, 83.4)

##--Base output path in directory--##
output_path = r"C:\Users\repooley\REP_PhD\NETCARE2015\data\processed\CurtainPlots\Meteorological\PTempLatitude"

##--Use f-string to save file with flight# appended--##
output_path = f"{output_path}\\{flight}_Temperature.png"
plt.savefig(output_path, dpi=600, bbox_inches='tight') 
plt.tight_layout()
plt.show()

########################
##--Diagnostic Plots--##
########################

##--Remove hashtags below to comment out this section--##
#'''

##--RH wrt water counts per bin data--##
RH_w_bin_counts, _, _, _ = binned_statistic_2d(clean_w_df['Latitude'], 
    clean_w_df['PTemp'], clean_w_df['Relative_Humidity_w'], statistic='count', bins=[common_lat_bin_edges, common_ptemp_bin_edges])
 
##--RH wrt ice counts per bin data--##
RH_i_bin_counts, _, _, _ = binned_statistic_2d(clean_i_df['Latitude'], 
    clean_i_df['PTemp'], clean_i_df['Relative_Humidity_i'], statistic='count', bins=[common_lat_bin_edges, common_ptemp_bin_edges])

##--Plotting--##

##--RH wrt water--##
fig1, ax1 = plt.subplots(figsize=(8, 6))

##--Make special color map where 0 values are white--##
new_cmap = plt.get_cmap('inferno')
##--Values under specified minimum will be white--##
new_cmap.set_under('w')

##--Use pcolormesh for the plot, set minimum value for viridis colors as 1--##
RH_w_diag_plot = ax1.pcolormesh(common_lat_bin_edges, common_ptemp_bin_edges, RH_w_bin_counts.T,  # Transpose to align correctly
    shading='auto', cmap=new_cmap, vmin=1, vmax=1250)

##--Add colorbar--##
cb = fig1.colorbar(RH_w_diag_plot, ax=ax1)
cb.minorticks_on()
cb.set_label('Number of Data Points', fontsize=12)

# Set axis labels
ax1.set_xlabel('Latitude (°)', fontsize=12)
ax1.set_ylabel('Potential Temperature \u0398 (K)', fontsize=12)
ax1.set_title(f"Relative Humidity wrt Water Counts per Bin - {flight.replace('Flight', 'Flight ')}")
#ax1.set_ylim(238, 301)
#ax1.set_xlim(82.4, 83.4)

##--Base output path in directory--##
output_path = r"C:\Users\repooley\REP_PhD\NETCARE2015\data\processed\CurtainPlots\Meteorological\PTempLatitude"

##--Use f-string to save file with flight# appended--##
output_path = f"{output_path}\\{flight}_RHwrtWater_diagnostic.png"
plt.savefig(output_path, dpi=600, bbox_inches='tight') 
plt.tight_layout()
plt.show()

##--RH wrt water--##
fig2, ax2 = plt.subplots(figsize=(8, 6))

##--Use pcolormesh for the plot, set minimum value for viridis colors as 1--##
RH_i_plot = ax2.pcolormesh(common_lat_bin_edges, common_ptemp_bin_edges, RH_i_bin_counts.T,  # Transpose to align correctly
    shading='auto', cmap=new_cmap, vmin=1, vmax=1250)

##--Add colorbar--##
cb = fig2.colorbar(RH_i_plot, ax=ax2)
cb.minorticks_on()
cb.set_label('Number of Data Points', fontsize=12)

# Set axis labels
ax2.set_xlabel('Latitude (°)', fontsize=12)
ax2.set_ylabel('Potential Temperature \u0398 (K)', fontsize=12)
ax2.set_title(f"Relative Humidity wrt Ice Counts per Bin - {flight.replace('Flight', 'Flight ')}")
#ax2.set_ylim(238, 301)
#ax2.set_xlim(79.5, 83.7)

##--Base output path in directory--##
output_path = r"C:\Users\repooley\REP_PhD\NETCARE2015\data\processed\CurtainPlots\Meteorological\PTempLatitude"

##--Use f-string to save file with flight# appended--##
output_path = f"{output_path}\\{flight}_RHwrtIce_diagnostic.png"
plt.savefig(output_path, dpi=600, bbox_inches='tight') 
plt.tight_layout()
plt.show()

#'''
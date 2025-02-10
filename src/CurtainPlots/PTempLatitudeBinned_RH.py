# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 14:42:01 2025

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

###########################
##--Create 2D histogram--##
###########################

##--Creates separate dfs to preserve data--##
w_df = pd.DataFrame({'PTemp': potential_temp, 'Latitude': latitude, 'Relative_Humidity_w': relative_humidity_w})
i_df = pd.DataFrame({'PTemp': potential_temp, 'Latitude': latitude, 'Relative_Humidity_i': relative_humidity_i})

##--Drop NaNs to prevent issues with potential_temp floats--##
clean_w_df = w_df.dropna()
clean_i_df = i_df.dropna()

##--Define number of bins here--##
num_bins_lat = 5
num_bins_ptemp = 15

##--Determine bin edges--##
w_lat_bin_edges = np.linspace(clean_w_df['Latitude'].min(), clean_w_df['Latitude'].max(), num_bins_lat + 1)
w_ptemp_bin_edges = np.linspace(clean_w_df['PTemp'].min(), clean_w_df['PTemp'].max(), num_bins_ptemp + 1)
i_lat_bin_edges = np.linspace(clean_i_df['Latitude'].min(), clean_i_df['Latitude'].max(), num_bins_lat +1)
i_ptemp_bin_edges = np.linspace(clean_i_df['PTemp'].min(), clean_i_df['PTemp'].max(), num_bins_ptemp +1)

##--Make 2d histogram and compute median RH in each bin--##
RH_w_bin_medians, RH_w_x_edges, RH_w_y_edges, _ = binned_statistic_2d(clean_w_df['Latitude'], 
    clean_w_df['PTemp'], clean_w_df['Relative_Humidity_w'], statistic='median', bins=[w_lat_bin_edges, w_ptemp_bin_edges])
RH_i_bin_medians, RH_i_x_edges, RH_i_y_edges, _ = binned_statistic_2d(clean_i_df['Latitude'], 
    clean_i_df['PTemp'], clean_i_df['Relative_Humidity_i'], statistic='median', bins=[i_lat_bin_edges, i_ptemp_bin_edges])

################
##--PLOTTING--##
################

##--RH wrt water--##
fig1, ax1 = plt.subplots(figsize=(8, 6))

##--Make special color map where 0 values are white--##
new_cmap = plt.get_cmap('viridis')
##--Values under specified minimum will be white--##
new_cmap.set_under('w')

##--Use pcolormesh for the plot, set minimum value for viridis colors as 1--##
RH_w_plot = ax1.pcolormesh(RH_w_x_edges, RH_w_y_edges, RH_w_bin_medians.T,  # Transpose to align correctly
    shading='auto', cmap=new_cmap, vmin=0, vmax=120)

##--Add colorbar--##
cb = fig1.colorbar(RH_w_plot, ax=ax1)
cb.minorticks_on()
cb.set_label('Relative Humidity', fontsize=12)

# Set axis labels
ax1.set_xlabel('Latitude (°)', fontsize=12)
ax1.set_ylabel('Potential Temperature \u0398 (K)', fontsize=12)
ax1.set_title(f"Relative Humidity wrt Water - {flight.replace('Flight', 'Flight ')}")
#ax1.set_ylim(238, 301)
#ax1.set_xlim(79.5, 83.7)

##--Base output path in directory--##
output_path = r"C:\Users\repooley\REP_PhD\NETCARE2015\data\processed\CurtainPlots\Meteorological\PTempLatitude"

##--Use f-string to save file with flight# appended--##
output_path = f"{output_path}\\{flight}_RHwrtWater.png"
plt.savefig(output_path, dpi=600, bbox_inches='tight') 
plt.tight_layout()
plt.show()

##--RH wrt water--##
fig2, ax2 = plt.subplots(figsize=(8, 6))

##--Use pcolormesh for the plot, set minimum value for viridis colors as 1--##
RH_i_plot = ax2.pcolormesh(RH_i_x_edges, RH_i_y_edges, RH_i_bin_medians.T,  # Transpose to align correctly
    shading='auto', cmap=new_cmap, vmin=0, vmax=120)

##--Add colorbar--##
cb = fig2.colorbar(RH_i_plot, ax=ax2)
cb.minorticks_on()
cb.set_label('Relative Humidity', fontsize=12)

# Set axis labels
ax2.set_xlabel('Latitude (°)', fontsize=12)
ax2.set_ylabel('Potential Temperature \u0398 (K)', fontsize=12)
ax2.set_title(f"Relative Humidity wrt Ice - {flight.replace('Flight', 'Flight ')}")
#ax2.set_ylim(238, 301)
#ax2.set_xlim(79.5, 83.7)

##--Base output path in directory--##
output_path = r"C:\Users\repooley\REP_PhD\NETCARE2015\data\processed\CurtainPlots\Meteorological\PTempLatitude"

##--Use f-string to save file with flight# appended--##
output_path = f"{output_path}\\{flight}_RHwrtIce.png"
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
    clean_w_df['PTemp'], clean_w_df['Relative_Humidity_w'], statistic='count', bins=[w_lat_bin_edges, w_ptemp_bin_edges])
 
##--RH wrt ice counts per bin data--##
RH_i_bin_counts, _, _, _ = binned_statistic_2d(clean_i_df['Latitude'], 
    clean_i_df['PTemp'], clean_i_df['Relative_Humidity_i'], statistic='count', bins=[i_lat_bin_edges, i_ptemp_bin_edges])

##--Plotting--##

##--RH wrt water--##
fig1, ax1 = plt.subplots(figsize=(8, 6))

##--Make special color map where 0 values are white--##
new_cmap = plt.get_cmap('inferno')
##--Values under specified minimum will be white--##
new_cmap.set_under('w')

##--Use pcolormesh for the plot, set minimum value for viridis colors as 1--##
RH_w_diag_plot = ax1.pcolormesh(RH_w_x_edges, RH_w_y_edges, RH_w_bin_counts.T,  # Transpose to align correctly
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
#ax1.set_xlim(79.5, 83.7)

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
RH_i_plot = ax2.pcolormesh(RH_i_x_edges, RH_i_y_edges, RH_i_bin_counts.T,  # Transpose to align correctly
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
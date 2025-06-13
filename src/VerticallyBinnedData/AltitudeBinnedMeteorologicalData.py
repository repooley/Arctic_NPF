# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 08:10:38 2024

@author: repooley
"""
import icartt
import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

#######################################
##--Open ICARTT Files and pull data--##
#######################################

##--Set the base directory to project folder--##
directory = r"C:\Users\repooley\REP_PhD\NETCARE2015\data"

##--Select flight (Flight1 thru Flight10)--##
flight = "Flight10" 

##--Define function that creates datasets from filenames--##
def find_files(directory, flight, partial_name):
    ##--flight data are stored in a folder called "raw"--##
    flight_dir = os.path.join(directory, "raw", flight)
    search_pattern = os.path.join(flight_dir, f"*{partial_name}*")
    return sorted(glob.glob(search_pattern))

##--Meterological data from AIMMS monitoring system--##
aimms = icartt.Dataset(find_files(directory, flight, "AIMMS_POLAR6")[0])
aimms_time =aimms.data['TimeWave']

##--Pulls data for each AIMMS variable--##
temperature = aimms.data["Temp"] # deg C
altitude = aimms.data['Alt']
min_alt = np.nanmin(altitude)
##--Humidity is converted to percent--##
probe_humidity = aimms.data['RH']*100
pressure = aimms.data['BP']

##--Pull H2O mixing data to calculate RH--##
H2O = icartt.Dataset(find_files(directory, flight, "H2O_POLAR6")[0])
H2O_probe = H2O.data['H2O_ppmv']

##--Ensure H2O data start/stop time is aligned with AIMMS--##
aimms_start = aimms_time.min()
aimms_end = aimms_time.max()

H2O_time = H2O.data['Time_UTC']
H2O_conc = H2O.data['H2O_ppmv']

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

##--Generate empty lists for humididy outputs--##
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
    
###############
##--BINNING--##
###############

##--Creates a Pandas dataframe with all variables--##
df = pd.DataFrame({'Altitude': altitude, 'Temperature': temperature_k, 'Potential_temp': potential_temp, 
                   'H2O_ppmv': H2O_conc_aligned, 'Probe_Humidity': probe_humidity, 
                   'Relative_Humidity_w': relative_humidity_w, 'Saturation_Humidity_w' : saturation_humidity_w, 
                   'Relative_Humidity_i': relative_humidity_i, 'Saturation_Humidity_i' : saturation_humidity_i})

##--Define number of bins here--##
num_bins = 80

# Compute the minimum and maximum altitude, ignoring NaNs
min_alt = df['Altitude'].min(skipna=True)
max_alt = df['Altitude'].max(skipna=True)

# Create bin edges from min_alt to max_alt
bin_edges = np.linspace(min_alt, max_alt, num_bins + 1)

##--Pandas 'cut' splits altitude data into specified number of bins--##
df['Altitude_bin'] = pd.cut(df['Altitude'], bins=bin_edges)

##--Group variables into each altitude bin--## 
##--Observed=false shows all bins, even empty ones--##
binned_df = df.groupby('Altitude_bin', observed=False).agg(
    
   ##--Aggregate data by mean, min, and max--##
    Temperature_avg=('Temperature', 'mean'),
    Temperature_min=('Temperature', 'min'),
    Temperature_max=('Temperature', 'max'),
    Temperature_25th=('Temperature', lambda x: x.quantile(0.25)),
    Temperature_75th=('Temperature', lambda x: x.quantile(0.75)),
    Potential_temp_avg=('Potential_temp', 'mean'),
    Potential_temp_min=('Potential_temp', 'min'),
    Potential_temp_max=('Potential_temp', 'max'),
    Potential_temp_25th=('Potential_temp', lambda x: x.quantile(0.25)),
    Potential_temp_75th=('Potential_temp', lambda x: x.quantile(0.75)),
    H2O_conc_avg=('H2O_ppmv', 'mean'),
    H2O_conc_min=('H2O_ppmv', 'min'),
    H2O_conc_max=('H2O_ppmv', 'max'),
    H2O_conc_25th=('H2O_ppmv', lambda x: x.quantile(0.25)),
    H2O_conc_75th=('H2O_ppmv', lambda x: x.quantile(0.75)),
    Probe_humidity_avg=('Probe_Humidity', 'mean'),
    Probe_humidity_min=('Probe_Humidity', 'min'),
    Probe_humidity_max=('Probe_Humidity', 'max'),
    Probe_humidity_25th=('Probe_Humidity', lambda x: x.quantile(0.25)),
    Probe_humidity_75th=('Probe_Humidity', lambda x: x.quantile(0.75)),    
    Rel_humidity_w_avg=('Relative_Humidity_w', 'mean'),
    Rel_humidity_w_min=('Relative_Humidity_w', 'min'),
    Rel_humidity_w_max=('Relative_Humidity_w', 'max'),
    Rel_humidity_w_25th=('Relative_Humidity_w', lambda x: x.quantile(0.25)),
    Rel_humidity_w_75th=('Relative_Humidity_w', lambda x: x.quantile(0.75)),
    Rel_humidity_i_avg=('Relative_Humidity_i', 'mean'),
    Rel_humidity_i_min=('Relative_Humidity_i', 'min'),
    Rel_humidity_i_max=('Relative_Humidity_i', 'max'),
    Rel_humidity_i_25th=('Relative_Humidity_i', lambda x: x.quantile(0.25)),
    Rel_humidity_i_75th=('Relative_Humidity_i', lambda x: x.quantile(0.75)),
    Altitude_center=('Altitude', 'mean')
    
##--Reset the index so Altitude_bin is just a column--##
).reset_index()

################
##--PLOTTING--##
################

##--Creates figure with 6 horizontally stacked subplots sharing a y-axis--##
fig, axs = plt.subplots(1, 6, figsize=(18, 6), sharey=True)

##--First subplot: Absolute Temperature vs Altitude--##
##--Averaged data in each bin is plotted against bin center--##
axs[0].plot(binned_df['Temperature_avg'], binned_df['Altitude_center'], color='#1c250c', label='Absolute Temperature')
##--Range is given by filling between data minimum and maximum for each bin--##
axs[0].fill_betweenx(binned_df['Altitude_center'], binned_df['Temperature_25th'], binned_df['Temperature_75th'], color='darkolivegreen', alpha=0.4)
axs[0].fill_betweenx(binned_df['Altitude_center'], binned_df['Temperature_min'], binned_df['Temperature_max'], color='darkolivegreen', alpha=0.2)
axs[0].set_xlabel('Absolute Temperature (K)')
axs[0].set_ylabel('Altitude (m)')
axs[0].set_title('Absolute Temperature')
#axs[0].set_xlim(-45, -16)

##--Second subplot: Altitude vs Potential Temperature--##
axs[1].plot(binned_df['Potential_temp_avg'], binned_df['Altitude_center'], color='#004242', label='Potential Temperature')
axs[1].fill_betweenx(binned_df['Altitude_center'], binned_df['Potential_temp_25th'], binned_df['Potential_temp_75th'], color='seagreen', alpha=0.4)
axs[1].fill_betweenx(binned_df['Altitude_center'], binned_df['Potential_temp_min'], binned_df['Potential_temp_max'], color='seagreen', alpha=0.2)
axs[1].set_xlabel('Potential Temperature (K)')
axs[1].set_title('Potential Temperature')

##--Third subplot: Water MR--##
axs[2].plot(binned_df['H2O_conc_avg'], binned_df['Altitude_center'], color='#002323', label='Probe RH')
axs[2].fill_betweenx(binned_df['Altitude_center'], binned_df['H2O_conc_25th'], binned_df['H2O_conc_75th'], color='teal', alpha=0.4)
axs[2].fill_betweenx(binned_df['Altitude_center'], binned_df['H2O_conc_min'], binned_df['H2O_conc_max'], color='teal', alpha=0.2)
axs[2].set_xlabel('ppmv Water')
axs[2].set_title('Water Mixing Ratio')
#axs[2].set_xlim(-45, -16)

##--Fourth subplot: RH wrt water from H2O data--##
axs[3].plot(binned_df['Rel_humidity_w_avg'], binned_df['Altitude_center'], color='darkslategray', label='Absolute Temperature')
axs[3].fill_betweenx(binned_df['Altitude_center'], binned_df['Rel_humidity_w_25th'], binned_df['Rel_humidity_w_75th'], color='cadetblue', alpha=0.7)
axs[3].fill_betweenx(binned_df['Altitude_center'], binned_df['Rel_humidity_w_min'], binned_df['Rel_humidity_w_max'], color='cadetblue', alpha=0.3)
axs[3].set_xlabel('Relative Humidity (%)')
axs[3].set_title('RH from MR WRT Water')
#axs[3].set_xlim(-45, -16)

##--Fifth subplot: RH wrt ice from H2O data--##
axs[4].plot(binned_df['Rel_humidity_i_avg'], binned_df['Altitude_center'], color='navy', label='Potential Temperature')
axs[4].fill_betweenx(binned_df['Altitude_center'], binned_df['Rel_humidity_i_25th'], binned_df['Rel_humidity_i_75th'], color='lightsteelblue', alpha=1)
axs[4].fill_betweenx(binned_df['Altitude_center'], binned_df['Rel_humidity_i_min'], binned_df['Rel_humidity_i_max'], color='lightsteelblue', alpha=0.4)
axs[4].set_xlabel('Relative Humidity (%)')
axs[4].set_title('RH from MR WRT Ice')

##--Sixth subplot: RH from probe--##
axs[5].plot(binned_df['Probe_humidity_avg'], binned_df['Altitude_center'], color='indigo', label='Probe RH', alpha=1)
axs[5].fill_betweenx(binned_df['Altitude_center'], binned_df['Probe_humidity_25th'], binned_df['Probe_humidity_75th'], color='rebeccapurple', alpha=0.4)
axs[5].fill_betweenx(binned_df['Altitude_center'], binned_df['Probe_humidity_min'], binned_df['Probe_humidity_max'], color='rebeccapurple', alpha=0.2)
axs[5].set_xlabel('Relative Humidity (%)')
axs[5].set_title('RH from Probe') 
#axs[5].set_xlim(0, 120)

##--Use f-string to embed flight # variable in plot title--##
plt.suptitle(f"Vertical Meteorological Profiles - {flight.replace('Flight', 'Flight ')}", fontsize=16)

##--Adjusts layout to prevent overlapping--## 
plt.tight_layout(rect=[0, 0, 1, 0.99])

##--Base output path in directory--##
output_path = r"C:\Users\repooley\REP_PhD\NETCARE2015\data\processed\AltitudeBinnedData\Meteorological"

##--Use f-string to save file with flight# appended--##
output_path = f"{output_path}\\{flight}"
plt.savefig(output_path, dpi=600, bbox_inches='tight') 

plt.show()
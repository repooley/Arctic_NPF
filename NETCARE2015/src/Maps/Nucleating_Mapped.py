# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 16:37:55 2025

@author: repooley
"""

import icartt
import os
import glob
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

#########################
##--Open ICARTT Files--##
#########################

##--Set the base directory to project folder--##
directory = r"C:\Users\repooley\REP_PhD\NETCARE2015\data"

##--Select flight (Flight1 thru Flight10)--##
flight = "Flight1" 

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

#################
##--Pull data--##
#################

##--AIMMS Data--##
altitude = aimms.data['Alt'] # in m
latitude = aimms.data['Lat'] # in degrees
longitude = aimms.data['Lon'] # in degrees
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

################
##--Plotting--##
################

##--Create the map with a North Polar Stereo projection and orient map so North America is at bottom--##
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=-90)})

##--Add land and ocean features--##
##--Physical specifies natural elements. 10m resolution is very fine--##
land = cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='black', 
                                    facecolor='lightgray')
ocean = cfeature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='none', 
                                     facecolor='lightblue')
##--Add features to plot ax--##
ax.add_feature(ocean)  # Ocean in light blue
ax.add_feature(land)   # Land in light gray

##--Add coastlines--##
#ax.coastlines()

##--First two values establish longitude (want all longitudes) and second specify latitudes--##
##--Fliglht 1: 0, 25, 76, 82
##--Flights2-7: -100, -68, 79, 84--##
##--Flights 8-10: -145, -120, 65, 72--##
ax.set_extent([0, 25, 76, 82], crs=ccrs.PlateCarree())

##--Plot flight track colored by altitude. Change Z-order so appears on top of map--##
track = ax.scatter(longitude, latitude, c=nuc_particles, cmap='viridis', transform=ccrs.PlateCarree(), s=10, zorder=3)

##--Add gridlines--##
gl = ax.gridlines(draw_labels=True)
gl.top_labels = True
gl.bottom_labels = True

##--Use f-string to embed flight # variable in plot title--##
plt.title(f"2.5-10 nm Particles Along Track - {flight.replace('Flight', 'Flight ')}", fontsize=16)

##--Add colorbar--##
cbar = plt.colorbar(track, ax=ax, orientation='vertical', shrink=0.7)
cbar.ax.tick_params(labelsize=16)
cbar.set_label(label="2.5-10 nm Particles $(Counts/cm^{3})$", fontsize=14)

plt.show()
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 13:55:03 2025

@author: repooley
"""

import icartt
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker

#########################
##--Open ICARTT Files--##
#########################

##--Set the base directory to project folder--##
directory = r"C:\Users\repooley\REP_PhD\NETCARE2015\data"

##--Select flight (Flight1 thru Flight10)--##
flight = "Flight7"

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
UHSAS_bins = pd.read_csv(r"C:\Users\repooley\REP_PhD\NETCARE2015\data\raw\NETCARE2015_UHSAS_bins.csv")

##--Make list of columns to pull, each named bin_x--##
##--Bins 1-13 not trustworthy. Trim to use bins 14-99 (>85 nm)--##
bin_num = [f'bin_{i}' for i in range(14, 99)]

##--Information for bins 14 thru 99--##
bin_center = UHSAS_bins['bin_avg'].iloc[14:99]
lower_bound = UHSAS_bins['lower_bound'].iloc[14:99]
upper_bound = UHSAS_bins['upper_bound'].iloc[14:99]

##--Put column names and content in a dictionary and then convert to a Pandas df--##
UHSAS_bins = pd.DataFrame({col: UHSAS.data[col] for col in bin_num})

##--Create new column names by rounding the bin center values to the nearest integer--##
new_col_names = bin_center.round().astype(int).tolist()

##--Rename the UHSAS_bins df columns to bin average values--##
UHSAS_bins.columns = new_col_names

##--Add time, total_num to UHSAS_bins df--##
UHSAS_bins.insert(0, 'Time', UHSAS_time)

##--Align UHSAS_bins time to AIMMS time--##
UHSAS_bins_aligned = UHSAS_bins.set_index('Time').reindex(aimms_time)

######################
##--Calc N(2.5-10)--##
######################

##--CPC data--##
CPC10 = icartt.Dataset(find_files(directory, flight, 'CPC3772')[0])
CPC3 = icartt.Dataset(find_files(directory, flight, 'CPC3776')[0])

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
CPC_df = pd.DataFrame({'Altitude': altitude, 'Latitude': latitude, 'CPC3_conc':CPC3_conc_STP, 'CPC10_conc': CPC10_conc_STP})

##--Calculate N3-10 particles--##
nuc_particles = (CPC_df['CPC3_conc'] - CPC_df['CPC10_conc'])

##--Change calculated particle counts less than zero to NaN--##
nuc_particles = np.where(nuc_particles >= 0, nuc_particles, np.nan)

# Smooth with rolling average (window size in seconds, adjust 'window' as needed)
window_size = 30  # e.g., 30 seconds
nuc_particles_smooth = pd.Series(nuc_particles).rolling(window=window_size, center=True, min_periods=1).mean()

################
##--Plotting--##
################

##--Numpy array expected by pcolormesh--##
uhsas_conc = UHSAS_bins_aligned.to_numpy().T  

##--Create meshgrids for x and y axes--##
time_mesh, diameter_mesh = np.meshgrid(aimms_time, bin_center)

##--Use pcolormesh which is more flexible than imshow--## 
fig, ax1 = plt.subplots(figsize=(12, 8))
c = ax1.pcolormesh(time_mesh, diameter_mesh, uhsas_conc, shading='auto', cmap='viridis')

# Labels and colorbar
cb = plt.colorbar(c, ax=ax1, location='bottom', pad=0.1, shrink=0.65)
cb.set_label(label='Normalized UHSAS Particle Concentration (dN/dlogDp) [cm⁻³]', fontsize=14)
cb.ax.tick_params(labelsize=14)

# Optional adjustments
ax1.set_title(f"UHSAS Time Series - {flight.replace('Flight', 'Flight ')}", fontsize=20, pad=20)
ax1.set_xlim([aimms_time.min(), aimms_time.max()])
#ax1.set_ylim([bin_center.min(), bin_center.max()])
ax1.set_yscale('log')
custom_ticks = [100, 200, 300, 400, 500, 600, 700, 800, 900]
# Apply custom ticks
ax1.yaxis.set_major_locator(ticker.FixedLocator(custom_ticks))

# Optional: format them normally (just numbers)
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
ax1.tick_params(axis='y', labelsize=14)
ax1.tick_params(axis='x', labelsize=14)

# Secondary y-axis for altitude
ax2 = ax1.twinx()
ax2.plot(aimms_time, nuc_particles_smooth, 'k', label='N(2.5-10)')
ax2.set_ylabel('N(2.5-10) Abundance (particles/cm³)', fontsize=14)
ax2.legend(loc='upper right', fontsize=14)
ax2.tick_params(axis='y', labelsize=14)
ax2.set_ylim(0, 2000)

ax1.set_xlabel('Time (seconds since midnight UTC)', fontsize=14)
ax1.set_ylabel('Log of Particle Diameter (nm)', fontsize=14)


plt.tight_layout()
plt.show()


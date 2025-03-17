# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 08:56:58 2025

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
flight = "Flight2"

##--Define function that creates datasets from filenames--##
def find_files(directory, flight, partial_name):
    ##--flight data are stored in a folder called "raw"--##
    flight_dir = os.path.join(directory, "raw", flight)
    search_pattern = os.path.join(flight_dir, f"*{partial_name}*")
    return sorted(glob.glob(search_pattern))

##--Meterological data from AIMMS monitoring system--##
aimms = icartt.Dataset(find_files(directory, flight, "AIMMS_POLAR6")[0])

##--OPC data--##
OPC = icartt.Dataset(find_files(directory, flight, 'OPC')[0])

#########################
##--Pull & align data--##
#########################

##--AIMMS Data--##
altitude = aimms.data['Alt'] # in m
latitude = aimms.data['Lat'] # in degrees
temperature = aimms.data['Temp'] + 273.15 # in K
pressure = aimms.data['BP'] # in pa
aimms_time =aimms.data['TimeWave'] # seconds since midnight

##--OPC Data--##
OPC_time = OPC.data['Time_UTC'] # seconds since midnight

##--Bin data are in a CSV file--##
OPC_bin_info = pd.read_csv(r"C:\Users\repooley\REP_PhD\NETCARE2015\data\raw\NETCARE2015_OPC_bins.csv")

##--Information for bins 1 thru 31--##
bin_center = OPC_bin_info['bin_avg'].iloc[0:31]
lower_bound = OPC_bin_info['lower_bound'].iloc[0:31]
upper_bound = OPC_bin_info['upper_bound'].iloc[0:31]

##--Make list of columns to pull, each named Channel_x--##
bin_num = [f'Channel_{i}' for i in range(1, 32)]

##--Put column names and content in a dictionary and then convert to a Pandas df--##
OPC_bins = pd.DataFrame({col: OPC.data[col] for col in bin_num})

##--Create new column names by rounding the bin center values to the nearest integer--##
new_col_names = bin_center.round().astype(int).tolist()

##--Rename the OPC_bins df columns to bin average values--##
OPC_bins.columns = new_col_names

##--Add time, total_num to OPC_bins df--##
OPC_bins.insert(0, 'Time', OPC_time)

##--Align OPC_bins time to AIMMS time--##
OPC_bins_aligned = OPC_bins.set_index('Time').reindex(aimms_time)

########################
##--Normalize Counts--##
########################

##--OPC samples every six seconds. Most rows are NaN--##
##--Forward-fill NaN values to propagate last valid reading--##
##--Limit forward filling to 5 NaN rows--##
OPC_bins_filled = OPC_bins_aligned.ffill(limit=5)

##--Calculate dlogDp for each bin in numpy array--##
dlogDp = np.log(upper_bound.values) - np.log(lower_bound.values)

##--Get only particle count data (excluding 'Time')--##
OPC_particle_counts = OPC_bins_filled.loc[:, new_col_names]

##--Normalize counts by dividing by dlogDp across all rows--##
OPC_dNdlogDp = OPC_bins_filled.divide(dlogDp, axis=1)

# convert to STP!

P_STP = 101325  # Pa
T_STP = 273.15  # K

##--Create empty list for CPC3 particles--##
OPC_conc_STP = []

for OPC, T, P in zip(OPC_dNdlogDp.values, temperature, pressure):
    if np.isnan(T) or np.isnan(P):
        ##--Append with NaN if any input is NaN--##
        OPC_conc_STP.append([np.nan]*len(OPC))
    else:
        ##--Perform conversion if all inputs are valid--##
        corrected_OPC = OPC * (P_STP / P) * (T / T_STP)
        OPC_conc_STP.append(corrected_OPC)

# Convert back to DataFrame with same columns and index
OPC_conc_STP = pd.DataFrame(OPC_conc_STP, columns=OPC_dNdlogDp.columns, index=OPC_dNdlogDp.index)
    
################
##--Plotting--##
################

##--Numpy array expected by pcolormesh--##
OPC_conc = OPC_conc_STP.to_numpy().T  

##--Create meshgrids for x and y axes--##
time_mesh, diameter_mesh = np.meshgrid(aimms_time, bin_center)

##--Use pcolormesh which is more flexible than imshow--## 
fig, ax1 = plt.subplots(figsize=(12, 8))
c = ax1.pcolormesh(time_mesh, diameter_mesh, OPC_conc, shading='auto', cmap='viridis')

# Labels and colorbar
cb = plt.colorbar(c, ax=ax1, location='bottom', pad=0.1, shrink=0.65)
cb.set_label(label='Normalized OPC Particle Concentration (dN/dlogDp) [cm⁻³]', fontsize=14)
cb.ax.tick_params(labelsize=14)

# Optional adjustments
ax1.set_title(f"OPC Time Series - {flight.replace('Flight', 'Flight ')}", fontsize=20, pad=20)
ax1.set_xlim([aimms_time.min(), aimms_time.max()])
#ax1.set_ylim([bin_center.min(), bin_center.max()])
ax1.set_yscale('log')
#custom_ticks = [100, 200, 300, 400, 500, 600, 700, 800, 900]
# Apply custom ticks
#ax1.yaxis.set_major_locator(ticker.FixedLocator(custom_ticks))

# Optional: format them normally (just numbers)
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
ax1.tick_params(axis='y', labelsize=14)
ax1.tick_params(axis='x', labelsize=14)
ax1.set_xlabel('Time (seconds since midnight UTC)', fontsize=14)
ax1.set_ylabel('Log of Particle Diameter (nm)', fontsize=14)


plt.tight_layout()
plt.show()



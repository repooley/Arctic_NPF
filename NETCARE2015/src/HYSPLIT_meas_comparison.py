# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 09:02:24 2025

@author: repooley
"""

import os
import glob
import math
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates

###################
##--User inputs--##
###################

##--Base directory--##
directory = r"C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data\raw"

##--HYSPLIT directory--##
hysplit = r"C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data\raw\HYSPLIT\data\trajectories\5min_averaged"

##--Choose which flights to analyze here!--##
##--FLIGHT1 HAS NO UHSAS FILE--##
flights_to_analyze = ["Flight2", "Flight3", "Flight4", "Flight5", "Flight6", "Flight7", "Flight8", "Flight9", "Flight10"]

##--Filter to above the polar dome?--##
above_dome = True

##--Base output path for figures in directory--##
output_path = r"C:\Users\repooley\REP_PhD\NETCARE2015\data\processed\ViolinPlots\Meteorological"

##--Read in file containing all co-occuring data with back trajectories--##
Netcare = pd.read_csv(r"C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data\raw\Netcare.csv")

##################
##--Pull Files--##
##################

##--Define a function that creates datasets from filenames--##
def find_files(directory, flight, folder, partial_name):
    search_pattern = os.path.join(directory, flight, folder, f"*{partial_name}*")
    return sorted(glob.glob(search_pattern))

##--Get timestamps where trajectories were initialized--##
##--Trajectories were initialized every 10 minutes from the Netcare file--##
flight_times = Netcare[Netcare['Flight_num'].isin(flights_to_analyze)]

start_utc = int(flight_times['Time_start'].min())
end_utc = int(flight_times['Time_start'].max())
UTCs = list(range(start_utc, end_utc +1, 300))

##--Subset Netcare to times in UTCs--##
netcare_subset = flight_times[flight_times['Time_start'].isin(UTCs)]

##########################
##--Group trajectories--##
##########################

##--Store measurements and trajectory conditions--##
traj_dfs = []
meas_dfs = []

for flight in flights_to_analyze:

    ##--Pull rows from netcare file for this flight--##
    flight_rows = netcare_subset[netcare_subset['Flight_num'] == flight].copy()
    
    ##--Define measurement start/end to make sure trajs are within the right times--##
    measurement_start = flight_rows['datetime'].min()
    measurement_end   = flight_rows['datetime'].max()

    ##--Populate list with measurement data pulled from the netcare file--##
    meas_dfs.append(flight_rows[['datetime', 'Temp', 'RH', 'Flight_num']])

    ##--HYSPLIT directory--##
    flight_dir = os.path.join(hysplit, flight)
    
    ##--Sort the HYSPLIT files in dir by time--##
    hysplit_files = sorted([f for f in os.listdir(flight_dir) if f.endswith(".txt")])

    ##--Empty list for flight-by-flight traj data--##
    flight_traj_data = []
    
    ##--Suggestion from GPT5 model--##
    ##--Loop through Netcare timestamps--##
    for idx, init_time in enumerate(flight_rows['datetime']):
        if idx >= len(hysplit_files):
            ##--Safety check: skip if there are fewer HYSPLIT files than Netcare rows--##
            continue

        ##--Pull the specific file within the sorted directory--##
        file = hysplit_files[idx]
        df = pd.read_csv(os.path.join(flight_dir, file), sep=r'\s+')
        df = df.rename(columns={'DATE': 'DAY'})
        df['YEAR'] = df['YEAR'].apply(lambda y: y + 2000)

        ##--Only use ensemble 1 as centroid--##
        ##--HYSPLIT file naming begins with 1, not 0--##
        init_row = df[df['TRAJ'] == 1].sort_values('HOUR').iloc[-1]
        perturbed = df[df['TRAJ'] != 1]

        ##--Append trajectory info using the exact Netcare timestamp--##
        flight_traj_data.append({'DateTime': init_time,  # exact initialization timestamp
            'Temp_centroid': init_row['AIR_TEMP'],
            'RH_centroid': init_row['RELHUMID'],
            'Temp_min': perturbed['AIR_TEMP'].min(),
            'Temp_max': perturbed['AIR_TEMP'].max(),
            'RH_min': perturbed['RELHUMID'].min(),
            'RH_max': perturbed['RELHUMID'].max(),
            'Flight_num': flight})

    ##--Append with trajectory info for each flight--##
    traj_dfs.append(pd.DataFrame(flight_traj_data))

##--Concatenate all flights, ignoring index--##
traj_df = pd.concat(traj_dfs, ignore_index=True)
meas_df = pd.concat(meas_dfs, ignore_index=True)

################
##--Plotting--##
################

##--Make sure in proper datetime format--##
meas_df['datetime'] = pd.to_datetime(meas_df['datetime'])
traj_df['DateTime'] = pd.to_datetime(traj_df['DateTime'])

n_flights = len(flights_to_analyze)
n_cols = 3

##--Figure with three columns--##
n_rows = math.ceil(n_flights / n_cols)

##--Make figsize dependent on number of rows--##
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

##--Flatten axes to a 1D array from 2D array for iterating--##
axes = axes.flatten()

##--Apply global parameters--##
plt.rcParams.update({'font.size': 12,
    'axes.titlesize': 14, 'axes.labelsize': 12, 'xtick.labelsize': 12,
    'ytick.labelsize': 12, 'legend.fontsize': 12})

##--Globally store handles and labels for universal legend--##
handles, labels = [], []

##--Make a plot iteration for each flight--##
for i, flight in enumerate(flights_to_analyze):
    
    ##--Separate y-axis for temp and RH, same x-axis--##
    ax1 = axes[i]
    ax2 = ax1.twinx()
    
    meas_sub = meas_df[meas_df['Flight_num'] == flight]
    traj_sub = traj_df[traj_df['Flight_num'] == flight]
    
    ##--Make perturbed trajectories shaded range between min/max values--##
    ##--Call handles--##
    h1 = ax1.fill_between(traj_sub['DateTime'], traj_sub['Temp_min'], traj_sub['Temp_max'],
                          color='tab:red', alpha=0.3, label='Trajectory Temp Range')
    h2 = ax2.fill_between(traj_sub['DateTime'], traj_sub['RH_min'], traj_sub['RH_max'],
                          color='tab:blue', alpha=0.3, label='Trajectory RH Range')

    ##--Mark the centroids within each trajectory range--##
    h5, = ax1.plot(traj_sub['DateTime'], traj_sub['Temp_centroid'], 'o', color='firebrick', alpha=1, label='Trajectory Centroid Temp')
    h6, = ax2.plot(traj_sub['DateTime'], traj_sub['RH_centroid'], 'o', color='royalblue', alpha=1, label='Trajectory Centroid RH')
    
    ##--Add the actual measured time series--##
    h3, = ax1.plot(meas_sub['datetime'], meas_sub['Temp'], color='firebrick', label='Measured Temp')
    h4, = ax2.plot(meas_sub['datetime'], meas_sub['RH'], color='royalblue', label='Measured RH')
    
    ##--Add a title and labels to each subplot--##
    ax1.set_title(f"{flight.replace('Flight', 'Flight ')}", fontsize=14)
    ax1.set_xlabel("UTC Hour", fontsize=12)
    ax1.set_ylabel("Temperature (K)", color='firebrick', fontsize=12)
    ax2.set_ylabel("Relative Humidity (%)", color='royalblue', fontsize=12)
    
    ##--Format each x-axis with the UTC time--##
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    ##--Rotate labels for better readability--##
    ##--Use setp to set properties of multiple artists--##
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=12)
    plt.setp(ax1.get_yticklabels(), fontsize=12)
    plt.setp(ax2.get_yticklabels(), fontsize=12)
    
    ##--Only collect legend handles once (from first subplot)--##
    if i == 0:
        handles = [h1, h2, h3, h4, h5, h6]
        labels = [h.get_label() for h in handles]

##--Hide any unused subplots (i +1) by turning off both the row and column--##
for j in range(i + 1, n_rows * n_cols):
    axes[j].axis('off')

##--Single legend at bottom center 
fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, fontsize=12)

##--Header--##
fig.suptitle("Measured vs. HYSPLIT Initialization Conditions Across Flights",
             fontsize=20, y=0.96)

##--Ensure layout can fit suptitle and legend--##
plt.tight_layout(rect=[0, 0.04, 1, 0.97])
plt.show()
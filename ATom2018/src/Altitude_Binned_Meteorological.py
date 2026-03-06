# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 13:46:22 2026

@author: repooley
"""


import icartt
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from datetime import date

###################
##--User inputs--##
###################

##--Set the base directory to project folder--##
directory = r"C:\Users\repooley\REP_PhD\Arctic_NPF\ATom2018\data"

##--The flight10 file also includes flight11--##
flights_to_analyze = ["Flight2", "Flight10", "Flight11", "Flight12"]

##--Define function that creates datasets from filenames--##
def find_files(directory, flight):
    ##--flight data are stored in a folder called "raw"--##
    flight_dir = os.path.join(directory, "raw", flight)
    search_pattern = os.path.join(flight_dir, "*.ict")
    return sorted(glob.glob(search_pattern))

all_dfs = []

##--Loop through each flight in the list--##
for flight in flights_to_analyze:
    
    #########################
    ##--Open ICARTT Files--##
    #########################

    dataset = icartt.Dataset(find_files(directory, flight)[0])

    ####################################
    ##--Assign date to flight number--##
    ####################################
    
    if flight=="Flight2":
        flight_date = date(2018, 4, 27)
    elif flight=="Flight10":
        flight_date = date(2018, 4, 17)
    elif flight=="Flight11": 
        flight_date = date(2018, 5, 18)
    elif flight=="Flight12":
        flight_date = date(2018, 5, 19)
    
    #################
    ##--Pull data--##
    #################
    
    ##--AIMMS Data--##
    altitude = dataset.data['G_ALT'] # in m (not sure if this is best one)
    temperature = dataset.data['T'] # in K
    pressure = dataset.data['P'] * 100 # in Pa
    RH = dataset.data['Relative_Humidity'] # wrt water, percent
    time =dataset.data['UTC_Start'] # seconds since midnight UTC
    
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
    
    ##--Place in dataframe--##
    
    df = pd.DataFrame({'altitude': altitude, 'temp': temperature, 'RH': RH})
    
    all_dfs.append(df)

################
##--PLOTTING--##
################

##--Define number of bins here--##
num_bins = 124

fig, axs = plt.subplots(1, 2, figsize=(6, 6), sharey=True)

##--Colormap - assign a color to each flight--##
cmap = plt.cm.viridis
n_flights = len(all_dfs)
colors = [cmap(i / (n_flights - 1)) for i in range(n_flights)]

##--Loop over flights--##
for i, (flight, df) in enumerate(zip(flights_to_analyze, all_dfs)):
    ##--Compute the minimum and maximum altitude, ignoring NaNs--##
    min_alt = df['altitude'].min(skipna=True)
    max_alt = df['altitude'].max(skipna=True)

    ##--Create bin edges from min_alt to max_alt--##
    bin_edges = np.linspace(min_alt, max_alt, num_bins + 1)

    ##--Pandas 'cut' splits altitude data into specified number of bins--##
    df['altitude_bin'] = pd.cut(df['altitude'], bins=bin_edges)

    ##--Group variables into each altitude bin--## 
    ##--Observed=false shows all bins, even empty ones--##
    binned_df = df.groupby('altitude_bin', observed=False).agg(
        
       ##--Aggregate data by mean, min, and max--##
        altitude_center=('altitude', 'median'), 
        temp_center=('temp', 'median'), 
        RH_center=('RH', 'median'), 

        ##--Reset the index so altitude_bin is just a column--##
    ).reset_index()

    axs[0].plot(binned_df["temp_center"], binned_df["altitude_center"],
                color=colors[i], label=f'{flight} ({flight_date})')
    
    axs[1].plot(binned_df["RH_center"], binned_df["altitude_center"],
                color=colors[i], label=f'{flight} ({flight_date})')
    
##--Subplot 1--##
axs[0].set_ylabel("Altitude (m)", fontsize=16)
axs[0].set_xlabel("Temperature (K)", fontsize=14)
axs[0].set_title("Temperature", fontsize=16)
#axs[0].set_xlim(-50, 2000)
axs[0].tick_params(axis='both', labelsize=12)

##--Subplot 2--##
axs[1].set_title("Relative Humidity", fontsize=16)
axs[1].set_xlabel("RH w.r.t. Water (%)", fontsize=14)
#axs[1].set_xlim(-50, 3400)
axs[1].tick_params(axis='both', labelsize=12)
axs[1].legend()

plt.suptitle("ATom 2018 Vertical Meteorological Profiles", fontsize=18)

#plt.tight_layout(rect=[0, 0.05, 1, 0.99])
plt.show()
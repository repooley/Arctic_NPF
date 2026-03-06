# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 13:52:24 2026

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
    O3 = dataset.data['O3_CL'] # ppbv
    NO = dataset.data['NO_CL'] # ppbv
    NO2 = dataset.data['NO2_CL'] # ppbv
    SO2 = dataset.data['SO2_CIT'] # pptv
    HNO3 = dataset.data['HNO3_CIT'] # pptv
    C2H5OOH = dataset.data['C2H5OOH_GMI'] # ppbv
    
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
    
    df = pd.DataFrame({'ptemp': potential_temp, 'O3': O3, 'NO': NO, 'NO2': NO2, 'SO2': SO2, 'HNO3': HNO3, 'C2H5OOH': C2H5OOH})
    
    ##--Drop rows where the potential temperature is above 310 K for comparison to other campaigns--##
    df = df[df['ptemp']<310]
    
    all_dfs.append(df)

################
##--PLOTTING--##
################

##--Define number of bins here--##
num_bins = 124

fig, axs = plt.subplots(1, 6, figsize=(18, 6), sharey=True)

##--Colormap - assign a color to each flight--##
cmap = plt.cm.viridis
n_flights = len(all_dfs)
colors = [cmap(i / (n_flights - 1)) for i in range(n_flights)]

##--Loop over flights--##
for i, (flight, df) in enumerate(zip(flights_to_analyze, all_dfs)):
    ##--Compute the minimum and maximum altitude, ignoring NaNs--##
    min_ptemp = df['ptemp'].min(skipna=True)
    max_ptemp = df['ptemp'].max(skipna=True)

    ##--Create bin edges from min_alt to max_alt--##
    bin_edges = np.linspace(min_ptemp, max_ptemp, num_bins + 1)

    ##--Pandas 'cut' splits altitude data into specified number of bins--##
    df['ptemp_bin'] = pd.cut(df['ptemp'], bins=bin_edges)

    ##--Group variables into each altitude bin--## 
    ##--Observed=false shows all bins, even empty ones--##
    binned_df = df.groupby('ptemp_bin', observed=False).agg(
        
       ##--Aggregate data by mean, min, and max--##
        ptemp_center=('ptemp', 'median'), 
        O3_center=('O3', 'median'), 
        NO_center=('NO', 'median'), 
        NO2_center=('NO2', 'median'),
        SO2_center=('SO2', 'median'),
        HNO3_center=('HNO3', 'median'), 
        C2H5OOH_center=('C2H5OOH', 'median')

        ##--Reset the index so ptemp_bin is just a column--##
    ).reset_index()

    axs[0].plot(binned_df["O3_center"], binned_df["ptemp_center"],
                color=colors[i], label=f'{flight} ({flight_date})')
    
    axs[1].plot(binned_df["NO_center"], binned_df["ptemp_center"],
                color=colors[i], label=f'{flight} ({flight_date})')
    
    axs[2].plot(binned_df["NO2_center"], binned_df["ptemp_center"],
                color=colors[i], label=f'{flight} ({flight_date})')
    
    axs[3].plot(binned_df["SO2_center"], binned_df["ptemp_center"],
                color=colors[i], label=f'{flight} ({flight_date})')
    
    axs[4].plot(binned_df["HNO3_center"], binned_df["ptemp_center"],
                color=colors[i], label=f'{flight} ({flight_date})')
    
    axs[5].plot(binned_df["C2H5OOH_center"], binned_df["ptemp_center"],
                color=colors[i], label=f'{flight} ({flight_date})')
    
##--Subplot 1--##
axs[0].set_ylabel("Potential Temperature (K)", fontsize=16)
axs[0].set_title("$O_{3}$", fontsize=16)
axs[0].set_xlabel('$O_{3}$ (ppbv)', fontsize=14)
#axs[0].set_xlim(-50, 2000)
axs[0].tick_params(axis='both', labelsize=12)

##--Subplot 2--##
axs[1].set_title("NO", fontsize=16)
axs[1].set_xlabel("NO (ppbv)", fontsize=14)
#axs[1].set_xlim(-50, 3400)
axs[1].tick_params(axis='both', labelsize=12)

##--Subplot 3--##
axs[2].set_title("$NO_{2}$", fontsize=16)
axs[2].set_xlabel("$NO_{2}$ (ppbv)", fontsize=14)
#axs[2].set_xlim(-50, 3400)
axs[2].tick_params(axis='both', labelsize=12)

##--Subplot 4--##
axs[3].set_title("$SO_{2}$", fontsize=16)
axs[3].set_xlabel("$SO_{2} (pptv)$", fontsize=14)
#axs[3].set_xlim(-50, 3400)
axs[3].tick_params(axis='both', labelsize=12)

##--Subplot 5--##
axs[4].set_title("$HNO_{3}$", fontsize=16)
axs[4].set_xlabel("$HNO_{3}$ (pptv)", fontsize=14)
#axs[4].set_xlim(-50, 3400)
axs[4].tick_params(axis='both', labelsize=12)

##--Subplot 6--##
axs[5].set_title("$C_{2}H_{5}OOH$", fontsize=16)
axs[5].set_xlabel("$C_{2}H_{5}OOH$ (ppbv)", fontsize=14)
#axs[5].set_xlim(-50, 3400)
axs[5].tick_params(axis='both', labelsize=12)

plt.legend()

plt.suptitle("ATom 2018 Vertical Trace Gas Profiles", fontsize=18)

#plt.tight_layout(rect=[0, 0.05, 1, 0.99])
plt.show()
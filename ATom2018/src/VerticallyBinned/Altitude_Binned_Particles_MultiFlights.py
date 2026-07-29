# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 15:13:16 2026

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

flights_to_analyze = ["Flight2", "Flight10", "Flight11", "Flight12"]

##--Define function that creates datasets from filenames--##
def find_files(directory, flight):
    ##--flight data are stored in a folder called "raw"--##
    flight_dir = os.path.join(directory, "raw", flight)
    search_pattern = os.path.join(flight_dir, "*.ict")
    return sorted(glob.glob(search_pattern))

##--Assign dates to the flights--##
flight_dates = {"Flight2":  date(2018, 4, 27),
    "Flight10":  date(2018, 5, 17),
    "Flight11": date(2018, 5, 18),
    "Flight12": date(2015, 5, 19)}

all_dfs = []

##--Loop through each flight in the list--##
for flight in flights_to_analyze:
    
    ##--Follow which flight is processing--##
    print(f"Processing {flight}...")
    
    #########################
    ##--Open ICARTT Files--##
    #########################

    dataset = icartt.Dataset(find_files(directory, flight)[0])

    #################
    ##--Pull data--##
    #################
    
    ##--AIMMS Data--##
    altitude = dataset.data['G_ALT'] # in m (not sure if this is best one)
    temperature = dataset.data['T'] # in K
    pressure = dataset.data['P'] # in hPa
    RH = dataset.data['Relative_Humidity'] # wrt water, percent
    time =dataset.data['UTC_Start'] # seconds since midnight UTC
    nucleating = dataset.data['N_nucl_AMP'] # num/cm^3 STP (2.7-12 nm)
    aitken = dataset.data['N_aitken_AMP'] # num/cm^3 STP (12-60 nm)
    
    ##--There are notable outliers in the nucleating data--##
    
    ##--First convert to a series for calc--##
    nucleating_series = pd.Series(nucleating)
    
    ##--REMOVE OUTLIERS above 99th percentile--##
    p = 0.99
    
    ##--Compute threshold for each UHSAS column--##
    nucleating_thresh = nucleating_series.quantile(p)
    
    ##--keep only rows where each bin is below its threshold--##
    nucleating_filtered = nucleating_series[nucleating_series.le(nucleating_thresh)]
    
    nucleating_filtered = nucleating_series.mask(
    nucleating_series <= nucleating_thresh)
    
    #############################
    ##--Calculate Uncertainty--##
    #############################
    
    ##--Place in dataframe--##
    
    df = pd.DataFrame({'Altitude': altitude, 'nucleating': nucleating, 'aitken': aitken})
    
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
    
    ####################################
    ##--Assign date to flight number--##
    ####################################

    flight_date = flight_dates[flight]  

    ##--Bin edges--##
    min_alt = df["Altitude"].min()
    max_alt = df["Altitude"].max()
    bin_edges = np.linspace(min_alt, max_alt, num_bins + 1)
    
    ##--Potential temperature bins--##
    df["Altitude_bin"] = pd.cut(df["Altitude"], bins=bin_edges)
    
    ##--Bin medians--##
    binned_df = df.groupby('Altitude_bin', observed=False).agg(
        
       ##--Aggregate data by mean, min, and max--##
        Altitude_center=('Altitude', 'median'), 
        nuc_particles_center=('nucleating', 'median'), 
        nuc_particles_min=('nucleating', 'min'),
        nuc_particles_max=('nucleating', 'max'), 
        nuc_particles_25th=('nucleating', lambda x: x.quantile(0.25)),
        nuc_particles_75th=('nucleating', lambda x: x.quantile(0.75)),
        aitken_particles_center=('aitken', 'median'), 
        aitken_particles_min=('aitken', 'min'),
        aitken_particles_max=('aitken', 'max'), 
        aitken_particles_25th=('aitken', lambda x: x.quantile(0.25)),
        aitken_particles_75th=('aitken', lambda x: x.quantile(0.75))

        ##--Reset the index so Altitude_bin is just a column--##
    ).reset_index() 
    
    axs[0].plot(binned_df["nuc_particles_center"], binned_df["Altitude_center"],
                color=colors[i], label=f'{flight} ({flight_date})')
    
    axs[1].plot(binned_df["aitken_particles_center"], binned_df["Altitude_center"],
                color=colors[i], label=f'{flight} ({flight_date})')
    
##--Subplot 1--##
axs[0].set_ylabel("Altitude (m)", fontsize=16)
axs[0].set_xlabel("Counts/cm³", fontsize=14)
axs[0].set_title("2.7-12 nm", fontsize=16)
#axs[0].set_xlim(-50, 2000)
axs[0].tick_params(axis='both', labelsize=12)

##--Subplot 2--##
axs[1].set_title("12-60 nm", fontsize=16)
axs[1].set_xlabel("Counts/cm³", fontsize=14)
#axs[1].set_xlim(-50, 3400)
axs[1].tick_params(axis='both', labelsize=12)
axs[1].legend()

plt.suptitle("ATom 2018 Vertical Particle Profiles", fontsize=18)

#plt.tight_layout(rect=[0, 0.05, 1, 0.99])
plt.show()


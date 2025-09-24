# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 16:03:43 2025

@author: repooley
"""

import icartt
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

###################
##--User inputs--##
###################

##--Set the base directory to project folder--##
directory = r"C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data\raw"

##--Choose which flights to analyze here!--##
flights_to_analyze = ["Flight2", "Flight3", "Flight4", "Flight5", "Flight6", 
                      "Flight7", "Flight8", "Flight9", "Flight10"]

##--Base output path in directory--##
output_path = r"C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data\processed\PTempBinnedData\Particle"

#%%
#########################
##--Open ICARTT Files--##
#########################

##--Define a function to find all flight data--##
def get_all_flights(directory):
    ##--flights are iteratively named Flight1, Flight2, etc--##
    raw_dir = os.path.join(directory)
    return [flight for flight in os.listdir(raw_dir) if 
            os.path.isdir(os.path.join(raw_dir, flight)) and flight.startswith("Flight")]
 
##--Define a function that creates datasets from filenames--##
def find_files(flight_dir, partial_name):
    search_pattern = os.path.join(flight_dir, f"*{partial_name}*")
    return sorted(glob.glob(search_pattern))

#%%
#################
##--Pull data--##
#################

##--Store processed data here: --##
CPC3_dfs = []
CPC10_dfs = []
nuc_dfs = []
ptemp_dfs = []
 
##--Loop through each flight, pulling and analyzing data--##
for flight in flights_to_analyze:
    ##--Follow which flight is processing--##
    print(f"Processing {flight}...")
    ##--Populate flight_dir established in above function--##
    flight_dir = os.path.join(directory, flight)
    ##--Pull meteorological data from AIMMS monitoring system--##
    aimms_files = find_files(flight_dir, "AIMMS_POLAR6")
    if aimms_files:
        aimms = icartt.Dataset(aimms_files[0])
    else:
        print(f"No AIMMS_POLAR6 file found for {flight}. Skipping...")
        continue  # Skip to the next flight if AIMMS file is missing
 
    ##--Pull CPC files--##
    CPC10_files = find_files(flight_dir, 'CPC3772')
    CPC3_files = find_files(flight_dir, 'CPC3776')
 
    if CPC10_files and CPC3_files:
        ##--Make variables containing all CPC dataset objects--##
        CPC10 = icartt.Dataset(CPC10_files[0])
        CPC3 = icartt.Dataset(CPC3_files[0])
    else:
        print(f"Missing CPC data for {flight}. Skipping...")
        continue
    
    ##--Pull UHSAS files--##
    UHSAS_files = find_files(flight_dir, "UHSAS")
    if UHSAS_files: 
        UHSAS = icartt.Dataset(UHSAS_files[0])
    else: 
        print(f"No UHSAS file found for {flight}. Skipping...")
        continue
    
    #########################
    ##--Pull & align data--##
    #########################
    
    ##--AIMMS Data--##
    altitude = aimms.data['Alt'] # in m
    latitude = aimms.data['Lat'] # in degrees
    temperature = aimms.data['Temp'] + 273.15 # in K
    pressure = aimms.data['BP'] # in pa
    aimms_time =aimms.data['TimeWave'] # seconds since midnight
    
    ##--Establish AIMMS start/stop times--##
    aimms_end = aimms_time.max()
    aimms_start = aimms_time.min()
     
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
    
    ######################
    ##--Calc N(2.5-10)--##
    ######################

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
    CPC_df = pd.DataFrame({'CPC3_conc':CPC3_conc_STP, 'CPC10_conc': CPC10_conc_STP})

    ##--Calculate N3-10 particles--##
    nuc_particles = (CPC_df['CPC3_conc'] - CPC_df['CPC10_conc'])

    ##--Change calculated particle counts less than zero to NaN--##
    nuc_particles = np.where(nuc_particles >= 0, nuc_particles, np.nan)
    
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
    
    #########################
    ##--Create dataframes--##
    #########################
    
    ##--Drop NaNs, done for individual datasets for data preservation--##
    CPC3_df = pd.DataFrame({'Ptemp': potential_temp, 'Latitude': latitude, 
                            'CPC3_conc': CPC3_conc_STP}).dropna()
    CPC10_df = pd.DataFrame({'Ptemp': potential_temp, 'Latitude': latitude, 
                            'CPC10_conc': CPC10_conc_STP}).dropna()
    nuc_df = pd.DataFrame({'Ptemp': potential_temp, 'Latitude': latitude, 
                           'nuc_particles': nuc_particles}).dropna()
    ptemp_df = pd.DataFrame({'Ptemp': potential_temp})

    ##--Store all processed data and ensure in numpy arrays--##
    CPC3_dfs.append(CPC3_df[['Ptemp', 'Latitude', 'CPC3_conc']])
    CPC10_dfs.append(CPC10_df[['Ptemp', 'Latitude', 'CPC10_conc']])
    nuc_dfs.append(nuc_df[['Ptemp', 'Latitude', 'nuc_particles']])
    ptemp_dfs.append(ptemp_df[['Ptemp']])

#%%
###############
##--BINNING--##
###############

# Loop through all datasets
for i, (cpc3_df, cpc10_df, nuc_df, ptemp_df) in enumerate(zip(CPC3_dfs, CPC10_dfs, nuc_dfs, ptemp_dfs)):

    print(f"Plotting dataset {i+1}...")

    # Recreate a combined dataframe for binning
    df = pd.concat([ptemp_df, cpc3_df.drop(columns="Ptemp"), 
                    cpc10_df.drop(columns="Ptemp"), 
                    nuc_df.drop(columns="Ptemp")], axis=1)
    
    # Set bin edges
    num_bins = 60

    # Bin edges
    min_ptemp = df['Ptemp'].min(skipna=True)
    max_ptemp = df['Ptemp'].max(skipna=True)
    bin_edges = np.linspace(min_ptemp, max_ptemp, num_bins + 1)

    df['PTemp_bin'] = pd.cut(df['Ptemp'], bins=bin_edges)

    # Group into bins
    binned_df = df.groupby('PTemp_bin', observed=False).agg(
        PTemp_center=('Ptemp', 'median'),
        CPC10_conc_center=('CPC10_conc', 'median'),
        CPC3_conc_center=('CPC3_conc', 'median')
    ).reset_index()
#%%

################
##--PLOTTING--##
################

# Create figure with 3 horizontally stacked subplots sharing y-axis
fig, axs = plt.subplots(1, 3, figsize=(9, 6), sharey=True)

# Choose a colormap for gradient coloring
cmap = plt.cm.viridis
n_flights = len(flights_to_analyze)
colors = [cmap(i / (n_flights - 1)) for i in range(n_flights)]

for i, flight in enumerate(flights_to_analyze):
    cpc3_df = CPC3_dfs[i]
    cpc10_df = CPC10_dfs[i]
    nuc_df = nuc_dfs[i]
    ptemp_df = ptemp_dfs[i]

    # Merge aligned dataframes (keep one Ptemp column)
    df = pd.concat([
        ptemp_df,
        cpc3_df.drop(columns="Ptemp"),
        cpc10_df.drop(columns="Ptemp"),
        nuc_df.drop(columns="Ptemp")
    ], axis=1)

    # Bin edges
    min_ptemp = df["Ptemp"].min(skipna=True)
    max_ptemp = df["Ptemp"].max(skipna=True)
    bin_edges = np.linspace(min_ptemp, max_ptemp, num_bins + 1)

    df["PTemp_bin"] = pd.cut(df["Ptemp"], bins=bin_edges)

    # Bin by potential temperature
    binned_df = df.groupby("PTemp_bin", observed=False).agg(
        PTemp_center=("Ptemp", "median"),
        CPC10_conc_center=("CPC10_conc", "median"),
        CPC3_conc_center=("CPC3_conc", "median"),
        nuc_particles_center=("nuc_particles", "median")
    ).reset_index()

    color = colors[i]

    # --- Subplot 1: CPC10 ---
    axs[0].plot(binned_df["CPC10_conc_center"], binned_df["PTemp_center"],
                label=flight, color=color)

    # --- Subplot 2: CPC3 ---
    axs[1].plot(binned_df["CPC3_conc_center"], binned_df["PTemp_center"],
                label=flight, color=color)

    # --- Subplot 3: Nucleating ---
    axs[2].plot(binned_df["nuc_particles_center"], binned_df["PTemp_center"],
                label=flight, color=color)

# ---- Format subplot 1 ----
axs[0].set_ylabel("Potential Temperature (K)", fontsize=16)
axs[0].set_xlabel("Counts/cm³", fontsize=14)
axs[0].set_title("N ≥ 10 nm", fontsize=16)
axs[0].set_xlim(-50, 2000)
axs[0].tick_params(axis='both', labelsize=12)
axs[0].axhline(y=285, color="k", linestyle="--", linewidth=1)
axs[0].axhline(y=299, color="k", linestyle="--", linewidth=1)

# Polar dome labels
x_text = axs[0].get_xlim()[0] + 1050
axs[0].text(x_text, 282, "Polar Dome", fontsize=11, color="k",
            verticalalignment="center", horizontalalignment="left")
axs[0].text(x_text, 288, "Marginal Dome", fontsize=11, color="k",
            verticalalignment="center", horizontalalignment="left")

# ---- Format subplot 2 ----
axs[1].set_title("N ≥ 2.5 nm", fontsize=16)
axs[1].set_xlabel("Counts/cm³", fontsize=14)
axs[1].set_xlim(-50, 3400)
axs[1].tick_params(axis='both', labelsize=12)
axs[1].axhline(y=285, color="k", linestyle="--", linewidth=1)
axs[1].axhline(y=299, color="k", linestyle="--", linewidth=1)

# ---- Format subplot 3 ----
axs[2].set_title("$N_{2.5-10}$", fontsize=16)
axs[2].set_xlabel("Counts/cm³", fontsize=14)
axs[2].tick_params(axis='both', labelsize=12)
axs[2].axhline(y=285, color="k", linestyle="--", linewidth=1)
axs[2].axhline(y=299, color="k", linestyle="--", linewidth=1)

# ---- Legend and title ----
axs[2].legend(loc="lower right", fontsize=12)

plt.suptitle("NETCARE 2015 Vertical Particle Profiles", fontsize=18)

plt.tight_layout(rect=[0, 0.05, 1, 0.99])
plt.show()
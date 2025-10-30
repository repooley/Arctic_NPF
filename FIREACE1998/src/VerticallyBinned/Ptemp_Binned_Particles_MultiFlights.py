# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 17:34:08 2025

@author: repooley
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

###################
##--User inputs--##
###################

##--Set the base directory to project folder--##
directory = r"C:\Users\repooley\REP_PhD\Arctic_NPF\FIREACE1998\data\raw"

##--Choose which flights to analyze here!--##
flights_to_analyze = ["Flight3", 
                      "Flight7", "Flight8", "Flight9", "Flight10", "Flight11", "Flight12",
                      "Flight13", "Flight14", "Flight15", "Flight16", "Flight17", "Flight18"]

##--Base output path in directory--##
output_path = r"C:\Users\repooley\REP_PhD\Arctic_NPF\FIREACE1998\data\processed"

PCASP_bins_path = r"C:\Users\repooley\REP_PhD\Arctic_NPF\FIREACE1998\data\raw\FIREACE1998_PCASP_bins.csv"

#%%
##################
##--Open Files--##
##################

##--Define a function to find all flight data--##
def get_all_flights(directory):
    ##--flights are iteratively named Flight1, Flight2, etc--##
    raw_dir = os.path.join(directory)
    return [flight for flight in os.listdir(raw_dir) if 
            os.path.isdir(os.path.join(raw_dir, flight)) and flight.startswith("Flight")]
 
##--Define a function that creates datasets from filenames--##
def find_files(directory, flight, partial_name):
    search_pattern = os.path.join(directory, flight, f"*{partial_name}*")
    return sorted(glob.glob(search_pattern))

#%%
#################
##--Pull data--##
#################

##--Store processed data here: --##
particle_dfs = []
averaged_dfs = []
 
##--Loop through each flight, pulling and analyzing data--##
for flight in flights_to_analyze:
    ##--Follow which flight is processing--##
    print(f"Processing {flight}...")
   
    ##--Pull csv file containing all data--##
    files = find_files(directory, flight, "FIREACE")
    
    ##--The 1 hz data is always the first file--##
    if files:
        data = pd.read_csv(files[0])
    else:
        print(f"No FIRE-ACE file found for {flight}. Skipping...")
        continue  # Skip to the next flight if FIRE-ACE file is missing
 
    
    ##--Pull data variables from file--##
    time = data['Time'] # HHMMSS UTC time
    pressure = data['Pressure'] * 100 # in Pa
    temperature = data['Temperature'] + 273.15 # in K
    RH = data['RH'] # percent wrt water
    altitude = data['Altitude'] # in m (agl?)
    latitude = data['Latitude'] # degrees
    longitude = data['Longitude'] # degrees

    ##--Particle data, 3 and 10 nm cutoffs, respectively--##
    CPC3_data = data['CN3025_corrected'] # Uncorrected data has a flow issue
    CPC10_data = data['CN7610']

        
    averaged_data = pd.read_csv(find_files(directory, flight, "FIREACE")[1])
    averaged_data = averaged_data.set_index('Time', drop=False)

    PCASP_bins = pd.read_csv(PCASP_bins_path)

    PCASP_data = averaged_data.iloc[:, 14:29] # select PCASP data

    ##--Add time, total_num to UHSAS_bins df--##
    PCASP_data.insert(0, 'Time', averaged_data['Time'])

    ##--Set time as the index for later alignment--##
    PCASP_data = PCASP_data.set_index('Time')

    ##--15 total bins--##
    PCASP_bin_num = [f'bin_{i}' for i in range(1, 16)]

    ##--Information for bins--##
    PCASP_bin_center = PCASP_bins['bin_avg']
    PCASP_lower_bound = PCASP_bins['lower_bound']
    PCASP_upper_bound = PCASP_bins['upper_bound']

    ##--Put column names and content in a dictionary and then convert to a Pandas df--##
    PCASP_df = pd.DataFrame({col: PCASP_data[col] for col in PCASP_bin_num})

    ##--Create new column names by rounding the bin center values to the nearest integer--##
    PCASP_new_col_names = PCASP_bin_center.round().astype(int).tolist()

    ##--Rename the PCASP_bins df columns to bin average values--##
    PCASP_data.columns = PCASP_new_col_names

    ##--Nans are denoted by -8888--##

    #%%
    ######################
    ##--Calc N(2.5-10)--##
    ######################

    ##--Convert to STP!--##
    P_STP = 101325  # Pa
    T_STP = 273.15  # K

    ##--Create empty list for CPC3 particles--##
    CPC3_conc_STP = []

    for CPC3, T, P in zip(CPC3_data, temperature, pressure):
        if np.isnan(CPC3) or np.isnan(T) or np.isnan(P):
            ##--Append with NaN if any input is NaN--##
            CPC3_conc_STP.append(np.nan)
        else:
            ##--Perform conversion if all inputs are valid--##
            CPC3_conversion = CPC3 * (P_STP / P) * (T / T_STP)
            CPC3_conc_STP.append(CPC3_conversion)
        
    ##--Create empty list for CPC10 particles--##
    CPC10_conc_STP = []

    for CPC10, T, P in zip(CPC10_data, temperature, pressure):
        if np.isnan(CPC10) or np.isnan(T) or np.isnan(P):
            ##--Append with NaN if any input is NaN--##
            CPC10_conc_STP.append(np.nan)
        else:
            ##--Perform conversion if all inputs are valid--##
            CPC10_conversion = CPC10 * (P_STP / P) * (T / T_STP)
            CPC10_conc_STP.append(CPC10_conversion)

    ##--Creates a Pandas dataframe for particle data--##
    df = pd.DataFrame({'Altitude': altitude, 'CPC3_conc':CPC3_conc_STP, 'CPC10_conc': CPC10_conc_STP})

    ##--Calculate N3-10 particles--##
    nuc_particles = (df['CPC3_conc'] - df['CPC10_conc'])

    ##--Change calculated particle counts less than zero to NaN--##
    nuc_particles = np.where(nuc_particles >= 0, nuc_particles, np.nan)

    ##--Add nucleating particles to df--##
    df['nuc_particles'] = nuc_particles
    
    #############################################
    ##--Normalize PCASP and averaged CPC Data--##
    #############################################

    ##--Calculate dlogDp for each bin in numpy array--##
    dlogDp = np.log(PCASP_upper_bound.values) - np.log(PCASP_lower_bound.values)

    ##--Get only particle count data (excluding 'Time')--##
    PCASP_particle_counts = PCASP_data.loc[:, PCASP_new_col_names]

    ##--Normalize counts by dividing by dlogDp across all rows--##
    PCASP_dNdlogDp = PCASP_data.divide(dlogDp, axis=1)

    ##--Convert to STP!--##
    P_STP = 101325  # Pa
    T_STP = 273.15  # K

    ##--Create empty list for PCASP particles--##
    PCASP_STP = []

    for PCASP, T, P in zip(PCASP_dNdlogDp.values, averaged_data['Temperature']+273.15, averaged_data['Pressure']*100):
        if np.isnan(T) or np.isnan(P):
            ##--Append with NaN if any input is NaN--##
            PCASP_STP.append([np.nan]*len(PCASP))
        else:
            ##--Perform conversion if all inputs are valid--##
            corrected_PCASP = PCASP * (P_STP / P) * (T / T_STP)
            PCASP_STP.append(corrected_PCASP)

    ##--Convert back to DataFrame with same columns and index--##
    PCASP_STP = pd.DataFrame(PCASP_STP, columns=PCASP_dNdlogDp.columns, index=PCASP_dNdlogDp.index)

    CPC_averaged_data = pd.DataFrame({'CPC3': averaged_data['CN3025'], 'CPC10': averaged_data['CN7610']}) # select PCASP data

    ##--Add time, total_num to UHSAS_bins df--##
    CPC_averaged_data.insert(0, 'Time', averaged_data['Time'])

    ##--Set time as the index for later alignment--##
    CPC_averaged_data = CPC_averaged_data.set_index('Time')
    
    ##--Calculate *averaged* nucleating particles--##
    n_3_10_averaged = (CPC_averaged_data['CPC3'] - CPC_averaged_data['CPC10'])
    
    ##--Change calculated particle counts less than zero to NaN--##
    n_3_10_averaged = np.where(n_3_10_averaged >= 0, n_3_10_averaged, np.nan)

    ##--Create empty list for PCASP particles--##
    n_3_10_averaged_STP = []

    for n_3_10, T, P in zip(n_3_10_averaged, averaged_data['Temperature']+273.15, averaged_data['Pressure']*100):
        if np.isnan(T) or np.isnan(P) or np.isnan(n_3_10):
            ##--Append with NaN if any input is NaN--##
            n_3_10_averaged_STP.append(np.nan)
        else:
            ##--Perform conversion if all inputs are valid--##
            corrected_n_3_10_averaged = n_3_10 * (P_STP / P) * (T / T_STP)
            n_3_10_averaged_STP.append(corrected_n_3_10_averaged)
                
    ##--Convert back to DataFrame with same columns and index--##
    n_3_10_averaged_STP = pd.DataFrame({'n_3_10_STP': n_3_10_averaged_STP}, index=CPC_averaged_data.index)
    
    ##--Make a separate dataframe for the averaged data--##
    df_averaged = pd.DataFrame({'Altitude': averaged_data['Altitude'], 
                                'Latitude': averaged_data['Latitude'], 
                                'Time': averaged_data['Time']})
    
    ##--Reindex df_averaged to time--##
    df_averaged = df_averaged.set_index('Time', drop=False)
    
    ##--Add PCASP data to the dataframe--##
    df_averaged = pd.concat([df_averaged, PCASP_STP, n_3_10_averaged_STP], axis=1)

    ######################
    ##--Calc N(10-130)--##
    ######################
    
    ##--Calculate the total in STP--##
    PCASP_total_STP = []
    
    for total, T, P in zip(averaged_data['PCTcon'], averaged_data['Temperature']+273.15, averaged_data['Pressure']*100):
        if np.isnan(T) or np.isnan(P) or np.isnan(total):
            ##--Append with NaN if any input is NaN--##
            PCASP_total_STP.append(np.nan)
        else:
            ##--Perform conversion if all inputs are valid--##
            corrected_total_averaged = total * (P_STP / P) * (T / T_STP)
            PCASP_total_STP.append(corrected_total_averaged)
    
    ##--Create df with UHSAS total counts--##
    PCASP_total = pd.DataFrame({'Time': averaged_data['Time'], 'Total_count': PCASP_total_STP})
    
    ##--Set time as the index for later alignment--##
    PCASP_total = PCASP_total.set_index('Time')
    
    ##--Add the PCASP total to the averaged df--##
    df_averaged = pd.concat([df_averaged, PCASP_total], axis=1)

    ##--Create df with CPC10 counts and set index to time--##
    CPC10_counts = pd.DataFrame({'Time':averaged_data['Time'], 'Counts':averaged_data['CN7610']}).set_index('Time')

    ##--Calculate particles below UHSAS lower cutoff--##
    n_10_130 = (averaged_data['CN7610'] - PCASP_total['Total_count'])

    ##--Change calculated particle counts less than zero to NaN--##
    n_10_130 = np.where(n_10_130 >= 0, n_10_130, np.nan)

    ##--Put N(10-130) bin center in a df--##
    n_10_130_center = pd.DataFrame([70])

    ##--Convert n_10_130 to a df--##
    n_10_130 = pd.DataFrame({'70': n_10_130, 'Time':averaged_data['Time']}).set_index('Time')
    
    df_averaged['n_10_130'] = n_10_130
    
    ##--Compute TOTAL counts from all size bins combined--##
    df_averaged['Total_particles_STP'] = (df_averaged['n_3_10_STP'].fillna(0) + 
    df_averaged['n_10_130'].fillna(0) + df_averaged['Total_count'].fillna(0))
    
    #######################################
    ##--Calculate potential temperature--##
    #######################################

    ##--Constants--##
    p_0 = 1E5 # Reference pressure in Pa (1000 hPa)
    k = 0.286 # Poisson constant for dry air

    ##--Generate empty list for potential temperature output--##
    potential_temp = []
    potential_temp_averaged = []

    ##--Calculate potential temperature from ambient temp & pressure--##
    for T, P in zip(temperature, pressure):
        p_t = T*(p_0/P)**k
        potential_temp.append(p_t)
    
    ##--Separate calculation for the averaged data--##
    for T, P in zip(averaged_data['Temperature']+273.15, averaged_data['Pressure']*100):
        p_t = T*(p_0/P)**k
        potential_temp_averaged.append(p_t)

    df['ptemp'] = potential_temp
    
    df_averaged['PTemp'] = potential_temp_averaged
    
    ##--Drop rows where ptemp is NaN--##
    df = df.dropna(subset=['ptemp'])
    df_averaged = df_averaged.dropna(subset=['PTemp'])
    
    ##--Drop rows where Latitude or ptemp are negative--##
    df = df[(df['ptemp'] >= 0)]
    df_averaged = df_averaged[(df_averaged['PTemp'] >= 0)]
    
    particle_dfs.append(df)
    
    averaged_dfs.append(df_averaged)


#%%

################
##--PLOTTING--##
################

##--NUCLEATING PARTICLES--##

num_bins = 30
all_ptemp = pd.concat([df["ptemp"] for df in particle_dfs])
min_ptemp = all_ptemp.min(skipna=True)
max_ptemp = all_ptemp.max(skipna=True)
bin_edges = np.linspace(min_ptemp, max_ptemp, num_bins + 1)

fig, axs = plt.subplots(1, 3, figsize=(9, 6), sharey=True)

cmap = plt.cm.cividis
n_flights = len(flights_to_analyze)
colors = [cmap(i / (n_flights - 1)) for i in range(n_flights)]

for i, flight in enumerate(flights_to_analyze):
    particle_df = particle_dfs[i].copy()


    particle_df['PTemp_bin'] = pd.cut(particle_df['ptemp'], bins=bin_edges)

    ##--Binning--##
    binned_df = particle_df.groupby('PTemp_bin', observed=False).agg(
        PTemp_center=('ptemp', 'median'),
        CPC10_conc_center=('CPC10_conc', 'median'),
        CPC3_conc_center=('CPC3_conc', 'median'),
        nuc_particles_center=('nuc_particles', 'median')
    ).reset_index()

    color = colors[i]

    ##--CPC 10--##
    axs[0].plot(binned_df["CPC10_conc_center"], binned_df["PTemp_center"],
                label=flight, color=color)

    ##--CPC 3--##
    axs[1].plot(binned_df["CPC3_conc_center"], binned_df["PTemp_center"],
                label=flight, color=color)

    ##--Nucleating--##
    axs[2].plot(binned_df["nuc_particles_center"], binned_df["PTemp_center"],
                label=flight, color=color)


axs[0].set_ylabel("Potential Temperature (K)", fontsize=16)
axs[0].set_xlabel("Counts/cm³", fontsize=14)
axs[0].set_title("N ≥ 10 nm", fontsize=16)
axs[0].set_xlim(-50, 2500)
axs[0].tick_params(axis='both', labelsize=11)
axs[0].axhline(y=285, color="k", linestyle="--", linewidth=1)
axs[0].axhline(y=299, color="k", linestyle="--", linewidth=1)

axs[1].set_title("N ≥ 2.5 nm", fontsize=16)
axs[1].set_xlabel("Counts/cm³", fontsize=14)
axs[1].set_xlim(-50, 3500)
axs[1].tick_params(axis='both', labelsize=11)
axs[1].axhline(y=285, color="k", linestyle="--", linewidth=1)
axs[1].axhline(y=299, color="k", linestyle="--", linewidth=1)

axs[2].set_title("$N_{2.5-10}$", fontsize=16)
axs[2].set_xlabel("Counts/cm³", fontsize=14)
axs[2].tick_params(axis='both', labelsize=11)
axs[2].axhline(y=285, color="k", linestyle="--", linewidth=1)
axs[2].axhline(y=299, color="k", linestyle="--", linewidth=1)

axs[2].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=12)

plt.suptitle("FIRE-ACE 1998 Vertical Particle Profiles", fontsize=18)

plt.tight_layout(rect=[0, 0.05, 1, 0.99]) 

##--PCASP TOTAL COUNT--##

num_bins = 30
all_ptemp = pd.concat([df_averaged["PTemp"] for df in averaged_dfs])
min_ptemp = all_ptemp.min(skipna=True)
max_ptemp = all_ptemp.max(skipna=True)
bin_edges = np.linspace(min_ptemp, max_ptemp, num_bins + 1)

fig, axs = plt.subplots(1, 1, figsize=(6, 6), sharey=True)

cmap = plt.cm.cividis
n_flights = len(flights_to_analyze)
colors = [cmap(i / (n_flights - 1)) for i in range(n_flights)]

for i, flight in enumerate(flights_to_analyze):
    averaged_df = averaged_dfs[i].copy()

    averaged_df['PTemp_bin'] = pd.cut(averaged_df['PTemp'], bins=bin_edges)

    binned_df = averaged_df.groupby('PTemp_bin', observed=False).agg(
        PTemp_center=('PTemp', 'median'),
        total_particles_center=('Total_particles_STP', 'median')
    ).reset_index()

    color = colors[i]

    ##--TOTAL--##
    axs.plot(binned_df["total_particles_center"], binned_df["PTemp_center"],
                label=flight, color=color)

axs.set_ylabel("Potential Temperature (K)", fontsize=16)
axs.set_xlabel("Counts/cm³", fontsize=14)
axs.set_title("Total particle counts", fontsize=16)
axs.set_xlim(-50, 2500)
axs.tick_params(axis='both', labelsize=11)
axs.axhline(y=285, color="k", linestyle="--", linewidth=1)
axs.axhline(y=299, color="k", linestyle="--", linewidth=1)

axs.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=12)

plt.suptitle("FIRE-ACE 1998 Vertical Particle Profiles", fontsize=18)

plt.tight_layout(rect=[0, 0.05, 1, 0.99])

plt.show()
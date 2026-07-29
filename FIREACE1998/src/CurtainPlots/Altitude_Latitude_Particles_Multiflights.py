# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 15:33:09 2026

@author: repooley
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
from scipy.stats import binned_statistic_2d
import cmcrameri as cm

###################
##--User inputs--##
###################

##--Set the base directory to project folder--##
directory = r"C:\Users\repooley\REP_PhD\Arctic_NPF\FIREACE1998\data\raw"

##--Choose which flights to analyze here!--##
flights_to_analyze = ["Flight3", 
                      "Flight7", "Flight8", "Flight9", "Flight10", "Flight11", "Flight12",
                      "Flight13", "Flight14", "Flight15", "Flight16", "Flight17", "Flight18"]

##--Set binning for PTemp and Latitude--##
##--Define number of bins here--##
num_bins_lat = 4
num_bins_alt = 8

##--Separate bin numbers for the averaged data--##
num_bins_lat_averaged = 6
num_bins_alt_averaged = 6

PCASP_bins_path = r"C:\Users\repooley\REP_PhD\Arctic_NPF\FIREACE1998\data\raw\FIREACE1998_PCASP_bins.csv"

##--Base output path in directory--##
output_path = r"C:\Users\repooley\REP_PhD\Arctic_NPF\FIREACE1998\data\processed"
 
##################
##--Open Files--##
##################

##--Define function that creates datasets from filenames--##
def find_files(directory, flight, partial_name):
    ##--flight data are stored in a folder called "raw"--##
    flight_dir = os.path.join(directory, flight)
    search_pattern = os.path.join(flight_dir, f"*{partial_name}*")
    return sorted(glob.glob(search_pattern))

##--Define a function to find all flight data--##
def get_all_flights(directory):
    ##--flights are iteratively named Flight1, Flight2, etc--##
    raw_dir = os.path.join(directory)
    return [flight for flight in os.listdir(raw_dir) if 
            os.path.isdir(os.path.join(raw_dir, flight)) and flight.startswith("Flight")]

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
    latitude = data['Latitude'] # degrees, there are some stray negative values...
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
    df = pd.DataFrame({'Altitude': altitude, 'Latitude': latitude, 'Longitude':longitude, 
                       'CPC3_conc':CPC3_conc_STP, 'CPC10_conc': CPC10_conc_STP})

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
    
    ##--Start df_averaged from full averaged_data to retain lat/time coverage--##
    df_averaged = averaged_data.copy()
    
    ##--Ensure time is the index--##
    df_averaged = df_averaged.set_index("Time", drop=False)
    
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
    
    ##--Drop rows where Latitude is NaN--##
    df = df.dropna(subset=['Latitude'])
    df_averaged = df_averaged.dropna(subset=['Latitude'])
    
    ##--Drop rows where Latitude is negative--##
    df = df[(df['Latitude'] >= 0)]
    df_averaged = df_averaged[(df_averaged['Latitude'] >= 0)]
    
    particle_dfs.append(df)
    
    averaged_dfs.append(df_averaged)

###########################
##--Prepare for Binning--##
###########################

##--Compute global min/max for all flights (non-averaged) --##
all_lats = np.concatenate([df["Latitude"].values for df in particle_dfs])
all_alts = np.concatenate([df["Altitude"].values for df in particle_dfs])

lat_min, lat_max = np.nanmin(all_lats), np.nanmax(all_lats)
alt_min, alt_max = np.nanmin(all_alts), np.nanmax(all_alts)

##--Common binning edges--##
common_lat_bin_edges = np.linspace(lat_min, lat_max, num_bins_lat + 1)
common_alt_bin_edges = np.linspace(alt_min, alt_max, num_bins_alt + 1)

##--Binning for CPC3 data--##
all_latitudes_CPC3 = np.concatenate([df["Latitude"].values for df in particle_dfs])
all_alts_CPC3 = np.concatenate([df["Altitude"].values for df in particle_dfs])
all_CPC3_concs = np.concatenate([df["CPC3_conc"].values for df in particle_dfs])

CPC3_bin_medians, _, _, _ = binned_statistic_2d(
    all_latitudes_CPC3, all_alts_CPC3, all_CPC3_concs,
    statistic="median", bins=[common_lat_bin_edges, common_alt_bin_edges])

##--Binning for CPC10 data--##
all_latitudes_CPC10 = np.concatenate([df["Latitude"].values for df in particle_dfs])
all_alts_CPC10 = np.concatenate([df["Altitude"].values for df in particle_dfs])
all_CPC10_concs = np.concatenate([df["CPC10_conc"].values for df in particle_dfs])

CPC10_bin_medians, _, _, _ = binned_statistic_2d(
    all_latitudes_CPC10, all_alts_CPC10, all_CPC10_concs,
    statistic="median", bins=[common_lat_bin_edges, common_alt_bin_edges])

##--Binning for nucleating particle data (non-averaged) --##
all_latitudes_nuc = np.concatenate([df["Latitude"].values for df in particle_dfs])
all_alts_nuc = np.concatenate([df["Altitude"].values for df in particle_dfs])
all_nuc_particles = np.concatenate([df["nuc_particles"].values for df in particle_dfs])

nuc_bin_medians, _, _, _ = binned_statistic_2d(
    all_latitudes_nuc, all_alts_nuc, all_nuc_particles,
    statistic="median", bins=[common_lat_bin_edges, common_alt_bin_edges])

##--Binning for averaged nucleating particle data --##
all_latitudes_nuc_averaged = np.concatenate([df["Latitude"].values for df in averaged_dfs])
all_alts_nuc_averaged = np.concatenate([df["Altitude"].values for df in averaged_dfs])
all_nuc_particles_averaged = np.concatenate([df["n_3_10_STP"].values for df in averaged_dfs])

nuc_bin_medians_averaged, _, _, _ = binned_statistic_2d(
    all_latitudes_nuc_averaged, all_alts_nuc_averaged, all_nuc_particles_averaged,
    statistic="median", bins=[common_lat_bin_edges, common_alt_bin_edges])

##--Binning for averaged data: n 10-130 --##
all_latitudes_n_10_130 = np.concatenate([df["Latitude"].values for df in averaged_dfs])
all_alts_n_10_130 = np.concatenate([df["Altitude"].values for df in averaged_dfs])
all_n_10_130 = np.concatenate([df["n_10_130"].values for df in averaged_dfs])

n_10_130_bin_medians, _, _, _ = binned_statistic_2d(
    all_latitudes_n_10_130, all_alts_n_10_130, all_n_10_130,
    statistic="median", bins=[common_lat_bin_edges, common_alt_bin_edges])

##--Binning for averaged data: total count --##
all_latitudes_total = np.concatenate([df["Latitude"].values for df in averaged_dfs])
all_alts_total = np.concatenate([df["Altitude"].values for df in averaged_dfs])
all_total = np.concatenate([df['Total_particles_STP'].values for df in averaged_dfs])

total_bin_medians, _, _, _ = binned_statistic_2d(
    all_latitudes_total, all_alts_total, all_total,
    statistic="median", bins=[common_lat_bin_edges, common_alt_bin_edges])

cmap = plt.get_cmap('viridis')

################
##--PLOTTING--##
################
 
def plot_curtain(bin_medians, x_edges, y_edges, vmin, vmax, title, cbar_label): #, output_path):
    fig, ax = plt.subplots(figsize=(6, 6))
 
    ##--Makecolor map where 0 values are white--##
    new_cmap = cmap
    new_cmap.set_under('w')
 
    ##--Plot the 2D data using pcolormesh--##
    mesh = ax.pcolormesh(x_edges, y_edges, bin_medians.T, shading="auto", cmap=new_cmap, vmin=vmin, vmax=vmax)
 
    ##--Add colorbar--##
    cb = fig.colorbar(mesh, ax=ax, orientation='horizontal', location='bottom', pad=0.15)
    cb.minorticks_on()
    cb.ax.tick_params(labelsize=18)
    cb.set_label(cbar_label, fontsize=18)

    ##--Set axis labels and title--##
    ax.set_xlabel("Latitude (°)", fontsize=18)
    ax.set_ylabel("Altitude (m)", fontsize=18)
    ax.tick_params(axis='both', labelsize=18)
    ax.set_title(title, fontsize=20)
    ax.set_ylim(1, 7100)
    ax.set_xlim(64, 86)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    #ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
 
    ##--Save the plot--##
    #plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.tight_layout()
    plt.show()
 
##--Plot for CPC3--##
plot_curtain(CPC3_bin_medians, common_lat_bin_edges, common_alt_bin_edges, vmin=1, vmax=2000,
    title="Particles >2.5 nm Abundance", cbar_label="Particles >2.5 nm $(Counts/cm^{3})$")
    #output_path=f"{output_path}\\CPC3/PTempLatitude/MultiFlights.png")

##--Plot for CPC10--##
plot_curtain(CPC10_bin_medians, common_lat_bin_edges, common_alt_bin_edges, vmin=1, vmax=2000,
    title="Particles >10 nm Abundance", cbar_label="Particles >10 nm $(Counts/cm^{3})$")
    #output_path=f"{output_path}\\CPC10/PTempLatitude/MultiFlights.png")
 
##--Plot for nucleating particles--##
plot_curtain(nuc_bin_medians, common_lat_bin_edges, common_alt_bin_edges, vmin=0, vmax=1000,
    title="2.5-10 nm Particle Abundance", cbar_label="2.5-10 nm Particles $(Counts/cm^{3})$")
    #output_path=f"{output_path}\\Nucleating/PTempLatitude/MultiFlights.png")

##--Plot for n_10_130--##
plot_curtain(n_10_130_bin_medians, common_lat_bin_edges, common_alt_bin_edges, vmin=1, vmax=2000,
    title='10-130 nm Particle Abundance', cbar_label='10-130 nm Particles $(Counts/cm^{3})$')

##--Plot for total count--##
plot_curtain(total_bin_medians, common_lat_bin_edges, common_alt_bin_edges, vmin=1, vmax=2000,
    title='Total Particle Abundance', cbar_label='All Particles $(Counts/cm^{3})$')

########################
##--Diagnostic Plots--##
########################

##--Remove hashtags below to comment out this section--##

##--Counts per bin for CPC3 data--##
CPC3_bin_counts, _, _, _ = binned_statistic_2d(all_latitudes_CPC3, all_alts_CPC3, all_CPC3_concs,
    statistic="count", bins=[common_lat_bin_edges, common_alt_bin_edges])
 
##--Counts per bin for CPC10 data--##
CPC10_bin_counts, _, _, _ = binned_statistic_2d(all_latitudes_CPC10, all_alts_CPC10, all_CPC10_concs,
    statistic="count", bins=[common_lat_bin_edges, common_alt_bin_edges])
 
##--Counts per bin for N3-10 particles--##
nuc_bin_counts, _, _, _ = binned_statistic_2d(all_latitudes_nuc, all_alts_nuc, all_nuc_particles,
    statistic="count", bins=[common_lat_bin_edges, common_alt_bin_edges])

##--Counts per bin for N10-89 particles--##
n_10_130_bin_counts, _, _, _ = binned_statistic_2d(all_latitudes_n_10_130, all_alts_n_10_130, all_n_10_130,
    statistic="count", bins=[common_lat_bin_edges, common_alt_bin_edges])

##--Counts per bin for N10-89 particles--##
total_bin_counts, _, _, _ = binned_statistic_2d(all_latitudes_total, all_alts_total, all_total,
    statistic="count", bins=[common_lat_bin_edges, common_alt_bin_edges])

##--Plotting--##

def plot_curtain(bin_counts, x_edges, y_edges, vmin, vmax, title, cbar_label):
    fig, ax = plt.subplots(figsize=(8, 6))
 
    ##--Set NaN values to white--##
    cmap = plt.get_cmap('viridis')
    cmap.set_under('w')
 
    ##--Plot the 2D data using pcolormesh--##
    mesh = ax.pcolormesh(x_edges, y_edges, bin_counts.T, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
 
    ##--Add colorbar--##
    cb = fig.colorbar(mesh, ax=ax)
    cb.minorticks_on()
    cb.ax.tick_params(labelsize=16)
    cb.set_label(cbar_label, fontsize=16)
    
    
    ##--Set axis labels and title--##
    ax.set_xlabel("Latitude (°)", fontsize=18)
    ax.set_ylabel("Altitude (m)", fontsize=18)
    ax.tick_params(axis='both', labelsize=18)
    ax.set_title(title, fontsize=20)
    #ax.set_ylim(238, 315)
    ax.set_xlim(65, 85)
 
    ##--Save the plot--##
    #plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.tight_layout()
    plt.show()
 
##--Plot for CPC3 counts--##
plot_curtain(CPC3_bin_counts, common_lat_bin_edges, common_alt_bin_edges, vmin=1, vmax=5500, 
    title="Particles >2.5 nm Data Point Counts", cbar_label="Number of Data Points")
    #output_path=f"{output_path}\\CPC3/PtempLatitude/MultiFlights_diagnostic.png")
 
##--Plot for CPC10 counts--##
plot_curtain(CPC10_bin_counts, common_lat_bin_edges, common_alt_bin_edges, vmin=1, vmax=6000,  
    title="Particles >10 nm Data Point Counts", cbar_label="Number of Data Points")
    #output_path=f"{output_path}\\CPC10/PtempLatitude/MultiFlights_diagnostic.png")
 
##--Plot for N3-10 counts--##
plot_curtain(nuc_bin_counts, common_lat_bin_edges, common_alt_bin_edges, vmin=1, vmax=4000,  
    title="2.5-10 nm Data Point Counts", cbar_label="Number of Data Points")
    #output_path=f"{output_path}\\Nucleating/PtempLatitude/MultiFlights_diagnostic.png")

##--Plot for N10-130 counts--##
plot_curtain(n_10_130_bin_counts, common_lat_bin_edges, common_alt_bin_edges, vmin=1, vmax=500,  
    title="10-130 nm Data Point Counts", cbar_label="Number of Data Points")
    #output_path=f"{output_path}\\N_10_89/PtempLatitude/MultiFlights_diagnostic.png")

##--Plot for total counts--##
plot_curtain(total_bin_counts, common_lat_bin_edges, common_alt_bin_edges, vmin=1, vmax=500,  
    title="Total Count Data Point Counts", cbar_label="Number of Data Points")
    #output_path=f"{output_path}\\N_10_89/PtempLatitude/MultiFlights_diagnostic.png")

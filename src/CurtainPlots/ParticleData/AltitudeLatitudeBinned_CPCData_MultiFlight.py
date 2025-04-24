# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 08:48:46 2025

@author: repooley
"""

import icartt
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from scipy.stats import binned_statistic_2d
 
###################
##--User inputs--##
###################

##--Set the base directory to project folder--##
directory = r"C:\Users\repooley\REP_PhD\NETCARE2015\data\raw"
 
##--Choose which flights to analyze here!--##
flights_to_analyze = ["Flight2", "Flight3", "Flight4", "Flight5", "Flight6", "Flight7", "Flight8", "Flight9", "Flight10"]

##--Set binning for PTemp and Latitude--##
num_bins_lat = 4
num_bins_alt = 12

# Add global output path here

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
 
##--Store processed data here: --##
CPC3_dfs = []
CPC10_dfs = []
nuc_dfs = []
 
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
        CPC10 = icartt.Dataset(CPC10_files[0])
        CPC3 = icartt.Dataset(CPC3_files[0])
    else:
        print(f"Missing CPC data for {flight}. Skipping...")
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
    ##--Convert to STP--##
    ######################
    
    ##--Constants--##
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
            
    #########################
    ##--Create dataframes--##
    #########################
    
    ##--Calculate N3-10 nucleating particles--##
    ##--Place required variables in a dataframe--##
    CPC_df = pd.DataFrame({'CPC3_conc': CPC3_conc_STP,'CPC10_conc': CPC10_conc_STP})
    nuc_particles = (CPC_df['CPC3_conc'] - CPC_df['CPC10_conc'])
    
    ##--Change values less than zero to NaN (instead of dropping) to ensure alignment--##
    nuc_particles = np.where(nuc_particles >= 0, nuc_particles, np.nan)
    
    ##--Drop NaNs, done for individual datasets for data preservation--##
    CPC3_df = pd.DataFrame({'Altitude': altitude, 'Latitude': latitude, 
                            'CPC3_conc': CPC3_conc_STP}).dropna()
    CPC10_df = pd.DataFrame({'Altitude': altitude, 'Latitude': latitude, 
                            'CPC10_conc': CPC10_conc_STP}).dropna()
    nuc_df = pd.DataFrame({'Altitude': altitude, 'Latitude': latitude, 
                           'nuc_particles': nuc_particles}).dropna()

    ##--Store all processed data and ensure in numpy arrays--##
    CPC3_dfs.append(CPC3_df[['Altitude', 'Latitude', 'CPC3_conc']])
    CPC10_dfs.append(CPC10_df[['Altitude', 'Latitude', 'CPC10_conc']])
    nuc_dfs.append(nuc_df[['Altitude', 'Latitude', 'nuc_particles']])

###########################
##--Prepare for Binning--##
###########################

##--Binning for CPC3 data--##
all_latitudes_CPC3 = np.concatenate([df["Latitude"].values for df in CPC3_dfs])
all_altitudes_CPC3 = np.concatenate([df["Altitude"].values for df in CPC3_dfs])
all_CPC3_concs = np.concatenate([df["CPC3_conc"].values for df in CPC3_dfs])
 
lat_bin_edges_CPC3 = np.linspace(all_latitudes_CPC3.min(), all_latitudes_CPC3.max(), num_bins_lat + 1)
alt_bin_edges_CPC3 = np.linspace(all_altitudes_CPC3.min(), all_altitudes_CPC3.max(), num_bins_alt + 1)
 
CPC3_bin_medians, _, _, _ = binned_statistic_2d(all_latitudes_CPC3, all_altitudes_CPC3, 
    all_CPC3_concs, statistic="mean", bins=[lat_bin_edges_CPC3, alt_bin_edges_CPC3])
 
##--Binning for CPC10 data--##
all_latitudes_CPC10 = np.concatenate([df["Latitude"].values for df in CPC10_dfs])
all_altitudes_CPC10 = np.concatenate([df["Altitude"].values for df in CPC10_dfs])
all_CPC10_concs = np.concatenate([df["CPC10_conc"].values for df in CPC10_dfs])
 
lat_bin_edges_CPC10 = np.linspace(all_latitudes_CPC10.min(), all_latitudes_CPC10.max(), num_bins_lat + 1)
alt_bin_edges_CPC10 = np.linspace(all_altitudes_CPC10.min(), all_altitudes_CPC10.max(), num_bins_alt + 1)
 
CPC10_bin_medians, _, _, _ = binned_statistic_2d(all_latitudes_CPC10, all_altitudes_CPC10, 
    all_CPC10_concs, statistic="mean", bins=[lat_bin_edges_CPC10, alt_bin_edges_CPC10])
 
##--Binning for nucleating particle data--##
all_latitudes_nuc = np.concatenate([df["Latitude"].values for df in nuc_dfs])
all_altitudes_nuc = np.concatenate([df["Altitude"].values for df in nuc_dfs])
all_nuc_particles = np.concatenate([df["nuc_particles"].values for df in nuc_dfs])
 
lat_bin_edges_nuc = np.linspace(all_latitudes_nuc.min(), all_latitudes_nuc.max(), num_bins_lat + 1)
alt_bin_edges_nuc = np.linspace(all_altitudes_nuc.min(), all_altitudes_nuc.max(), num_bins_alt + 1)
 
nuc_bin_medians, _, _, _ = binned_statistic_2d(all_latitudes_nuc, all_altitudes_nuc, 
    all_nuc_particles, statistic="mean", bins=[lat_bin_edges_nuc, alt_bin_edges_nuc])
 
################
##--PLOTTING--##
################
 
def plot_curtain(bin_medians, x_edges, y_edges, vmin, vmax, title, cbar_label, output_path):
    fig, ax = plt.subplots(figsize=(8, 6))
 
    ##--Make special color map where 0 values are white--##
    new_cmap = plt.get_cmap('viridis')
    new_cmap.set_under('w')
 
    ##--Plot the 2D data using pcolormesh--##
    mesh = ax.pcolormesh(x_edges, y_edges, bin_medians.T, shading="auto", cmap=new_cmap, vmin=vmin, vmax=vmax)
 
    ##--Add colorbar--##
    cb = fig.colorbar(mesh, ax=ax)
    cb.minorticks_on()
    cb.set_label(cbar_label, fontsize=12)
 
    ##--Set axis labels and title--##
    ax.set_xlabel("Latitude (°)", fontsize=12)
    ax.set_ylabel("Altitude (m)", fontsize=12)
    ax.set_title(title)
    ax.set_ylim(0, 6250)
    #ax.set_xlim(79.5, 83.7)
 
    ##--Save the plot--##
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.tight_layout()
    plt.show()
 
##--Plot for CPC3--##
plot_curtain(CPC3_bin_medians, lat_bin_edges_CPC3, alt_bin_edges_CPC3, vmin=0, vmax=2250,
    title="Particles >2.5 nm Abundance", cbar_label="Particles >2.5 nm $(Counts/cm^{3})$",
    output_path=r"C:\Users\repooley\REP_PhD\NETCARE2015\data\processed\CurtainPlots\CPC3\AltitudeLatitude\MultiFlights.png")

##--Plot for CPC10--##
plot_curtain(CPC10_bin_medians, lat_bin_edges_CPC10, alt_bin_edges_CPC10, vmin=0, vmax=2000,
    title="Particles >10 nm Abundance", cbar_label="Particles >10 nm $(Counts/cm^{3})$",
    output_path=r"C:\Users\repooley\REP_PhD\NETCARE2015\data\processed\CurtainPlots\CPC10\AltitudeLatitude\MultiFlights.png")
 
##--Plot for nucleating particles--##
plot_curtain(nuc_bin_medians, lat_bin_edges_nuc, alt_bin_edges_nuc, vmin=0, vmax=1100,
    title="2.5-10 nm Particle Abundance", cbar_label="2.5-10 nm Particles $(Counts/cm^{3})$",
    output_path=r"C:\Users\repooley\REP_PhD\NETCARE2015\data\processed\CurtainPlots\Nucleating\AltitudeLatitude\MultiFlights.png")

########################
##--Diagnostic Plots--##
########################

##--Remove hashtags below to comment out this section--##
#'''

##--Counts per bin for CPC3 data--##
CPC3_bin_counts, _, _, _ = binned_statistic_2d(all_latitudes_CPC3, all_altitudes_CPC3, all_CPC3_concs,
    statistic="count", bins=[lat_bin_edges_CPC3, alt_bin_edges_CPC3])
 
##--Counts per bin for CPC10 data--##
CPC10_bin_counts, _, _, _ = binned_statistic_2d(all_latitudes_CPC10, all_altitudes_CPC10, all_CPC10_concs,
    statistic="count", bins=[lat_bin_edges_CPC10, alt_bin_edges_CPC10])
 
##--Counts per bin for N3-10 particles--##
nuc_bin_counts, _, _, _ = binned_statistic_2d(all_latitudes_nuc, all_altitudes_nuc, all_nuc_particles,
    statistic="count", bins=[lat_bin_edges_nuc, alt_bin_edges_nuc])

##--Plotting--##

def plot_curtain(bin_counts, x_edges, y_edges, vmin, vmax, title, cbar_label, output_path):
    fig, ax = plt.subplots(figsize=(8, 6))
 
    ##--Set NaN values to white--##
    cmap = plt.get_cmap('inferno')
    cmap.set_under('w')
 
    ##--Plot the 2D data using pcolormesh--##
    mesh = ax.pcolormesh(x_edges, y_edges, bin_counts.T, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
 
    ##--Add colorbar--##
    cb = fig.colorbar(mesh, ax=ax)
    cb.minorticks_on()
    cb.set_label(cbar_label, fontsize=12)

    ##--Set axis labels and title--##
    ax.set_xlabel("Latitude (°)", fontsize=12)
    ax.set_ylabel("Altitude (m)", fontsize=12)
    ax.set_title(title)
    ax.set_ylim(0, 6250)
    #ax.set_xlim(79.5, 83.7)
 
    ##--Save the plot--##
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.tight_layout()
    plt.show()
 
##--Plot for CPC3 counts--##
plot_curtain(CPC3_bin_counts, lat_bin_edges_CPC3, alt_bin_edges_CPC3, vmin=1, vmax=6000, 
    title="Particles >2.5 nm Data Point Counts", cbar_label="Number of Data Points",
    output_path=r"C:\Users\repooley\REP_PhD\NETCARE2015\data\processed\CurtainPlots\CPC3\AltitudeLatitude\MultiFlights_diagnostic.png")
 
##--Plot for CPC10 counts--##
plot_curtain(CPC10_bin_counts, lat_bin_edges_CPC10, alt_bin_edges_CPC10, vmin=1, vmax=7500,  
    title="Particles >10 nm Data Point Counts", cbar_label="Number of Data Points",
    output_path=r"C:\Users\repooley\REP_PhD\NETCARE2015\data\processed\CurtainPlots\CPC10\AltitudeLatitude\MultiFlights_diagnostic.png")
 
##--Plot for N3-10 counts--##
plot_curtain(nuc_bin_counts, lat_bin_edges_nuc, alt_bin_edges_nuc, vmin=1, vmax=3500,  
    title="2.5-10 nm Data Point Counts", cbar_label="Number of Data Points",
    output_path=r"C:\Users\repooley\REP_PhD\NETCARE2015\data\processed\CurtainPlots\Nucleating\AltitudeLatitude\MultiFlights_diagnostic.png")

#'''
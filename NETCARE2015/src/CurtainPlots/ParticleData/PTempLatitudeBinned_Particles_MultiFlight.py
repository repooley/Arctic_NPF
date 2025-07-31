# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 12:54:27 2025

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
##--Define number of bins here--##
num_bins_lat = 10
num_bins_ptemp = 10

##--UHSAS bins--##
bins_filepath = r"C:\Users\repooley\REP_PhD\NETCARE2015\data\raw\NETCARE2015_UHSAS_bins.csv"

##--Base output path for figures in directory--##
output_path = r"C:\Users\repooley\REP_PhD\NETCARE2015\data\processed\CurtainPlots"
 
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
SP2_dfs = []
CPC3_dfs = []
CPC10_dfs = []
nuc_dfs = []
grow_dfs = []
 
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
    
    ##--Black carbon data from SP2--##
    SP2_files = find_files(flight_dir, "SP2_Polar6")
    if SP2_files: 
        SP2 = icartt.Dataset(SP2_files[0])
    else: 
        print(f"No SP2 file found for {flight}. Skipping...")
        continue
 
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
    
    ##--Black carbon--##
    BC_count = SP2.data['BC_numb_concSTP'] # in STP

    ##--Handle black carbon data with different start/stop times than AIMMS--##
    BC_time = SP2.data['Time_UTC']

    ##--Trim CO data if it starts before AIMMS--##
    if BC_time.min() < aimms_start:
        mask_start = BC_time >= aimms_start
        BC_time = BC_time[mask_start]
        BC_count = BC_count[mask_start]
        
    ##--Append CO data with NaNs if it ends before AIMMS--##
    if BC_time.max() < aimms_end: 
        missing_times = np.arange(BC_time.max()+1, aimms_end +1)
        BC_time = np.concatenate([BC_time, missing_times])
        BC_count = np.concatenate([BC_count, [np.nan]*len(missing_times)])

    ##--Create a DataFrame for BC data and reindex to AIMMS time, setting non-overlapping times to nan--##
    BC_df = pd.DataFrame({'Time_UTC': BC_time, 'BC_count': BC_count})
    BC_aligned = BC_df.set_index('Time_UTC').reindex(aimms_time)
    BC_aligned['BC_count']= BC_aligned['BC_count'].where(BC_aligned.index.isin(aimms_time), np.nan)
    BC_count_aligned = BC_aligned['BC_count']
    
    ##--Bin data are in a CSV file--##
    UHSAS_bins = pd.read_csv(bins_filepath)
    
    ##--USHAS Data--##
    UHSAS_time = UHSAS.data['time'] # seconds since midnight
    ##--Total count is computed for N > 85 nm--##
    UHSAS_total_num = UHSAS.data['total_number_conc'] # particles/cm^3

    ##--Make list of columns to pull, each named bin_x--##
    ##--Bins 1-13 not trustworthy. Bins 76-99 overlap with OPC, discard--##
    ##--Trim to use bins 14-76 (500>85 nm)--##
    UHSAS_bin_num = [f'bin_{i}' for i in range(14, 75)]

    ##--Information for bins 14 thru 99--##
    UHSAS_bin_center = UHSAS_bins['bin_avg'].iloc[14:75]
    UHSAS_lower_bound = UHSAS_bins['lower_bound'].iloc[14:75]
    UHSAS_upper_bound = UHSAS_bins['upper_bound'].iloc[14:75]

    ##--Put column names and content in a dictionary and then convert to a Pandas df--##
    UHSAS_bins = pd.DataFrame({col: UHSAS.data[col] for col in UHSAS_bin_num})

    ##--Create new column names by rounding the bin center values to the nearest integer--##
    UHSAS_new_col_names = UHSAS_bin_center.round().astype(int).tolist()

    ##--Rename the UHSAS_bins df columns to bin average values--##
    UHSAS_bins.columns = UHSAS_new_col_names

    ##--Add time, total_num to UHSAS_bins df--##
    UHSAS_bins.insert(0, 'Time', UHSAS_time)

    ##--Align UHSAS_bins time to AIMMS time--##
    UHSAS_bins_aligned = UHSAS_bins.set_index('Time').reindex(aimms_time)
    
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

    #####################
    ##--Calc N(10-89)--##
    #####################

    ##--Create df with UHSAS total counts--##
    UHSAS_total = pd.DataFrame({'Time': UHSAS_time, 'Total_count': UHSAS_total_num})

    ##--Reindex UHSAS_total df to AIMMS time--##
    UHSAS_total_aligned = UHSAS_total.set_index('Time').reindex(aimms_time)

    ##--Create df with CPC10 counts and set index to time--##
    CPC10_counts = pd.DataFrame({'Time':aimms_time, 'Counts':CPC10_conc_STP}).set_index('Time')

    ##--Calculate particles below UHSAS lower cutoff--##
    n_10_89 = (CPC10_counts['Counts'] - UHSAS_total_aligned['Total_count'])

    ##--Change calculated particle counts less than zero to NaN--##
    n_10_89 = np.where(n_10_89 >= 0, n_10_89, np.nan)
    
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
    SP2_df = pd.DataFrame({'Ptemp': potential_temp, 'Latitude': latitude, 
                            'BC_count': BC_count_aligned}).dropna()
    CPC3_df = pd.DataFrame({'Ptemp': potential_temp, 'Latitude': latitude, 
                            'CPC3_conc': CPC3_conc_STP}).dropna()
    CPC10_df = pd.DataFrame({'Ptemp': potential_temp, 'Latitude': latitude, 
                            'CPC10_conc': CPC10_conc_STP}).dropna()
    nuc_df = pd.DataFrame({'Ptemp': potential_temp, 'Latitude': latitude, 
                           'nuc_particles': nuc_particles}).dropna()
    ##--Calling n 10-89 'growth'--##
    grow_df = pd.DataFrame({'Ptemp': potential_temp, 'Latitude': latitude, 
                           'n_10_89': n_10_89}).dropna()

    ##--Store all processed data and ensure in numpy arrays--##
    SP2_dfs.append(SP2_df[['Ptemp', 'Latitude', 'BC_count']])
    CPC3_dfs.append(CPC3_df[['Ptemp', 'Latitude', 'CPC3_conc']])
    CPC10_dfs.append(CPC10_df[['Ptemp', 'Latitude', 'CPC10_conc']])
    nuc_dfs.append(nuc_df[['Ptemp', 'Latitude', 'nuc_particles']])
    grow_dfs.append(grow_df[['Ptemp', 'Latitude', 'n_10_89']])

###########################
##--Prepare for Binning--##
###########################

##--Binning for rBC data--##
all_latitudes_BC = np.concatenate([df["Latitude"].values for df in SP2_dfs])
all_ptemps_BC = np.concatenate([df["Ptemp"].values for df in SP2_dfs])
all_BC_counts = np.concatenate([df["BC_count"].values for df in SP2_dfs])
 
lat_bin_edges_BC = np.linspace(all_latitudes_BC.min(), all_latitudes_BC.max(), num_bins_lat + 1)
ptemp_bin_edges_BC = np.linspace(all_ptemps_BC.min(), all_ptemps_BC.max(), num_bins_ptemp + 1)
 
BC_bin_medians, _, _, _ = binned_statistic_2d(all_latitudes_BC, all_ptemps_BC, 
        all_BC_counts, statistic="mean", bins=[lat_bin_edges_BC, ptemp_bin_edges_BC])

##--Binning for CPC3 data--##
all_latitudes_CPC3 = np.concatenate([df["Latitude"].values for df in CPC3_dfs])
all_ptemps_CPC3 = np.concatenate([df["Ptemp"].values for df in CPC3_dfs])
all_CPC3_concs = np.concatenate([df["CPC3_conc"].values for df in CPC3_dfs])
 
lat_bin_edges_CPC3 = np.linspace(all_latitudes_CPC3.min(), all_latitudes_CPC3.max(), num_bins_lat + 1)
ptemp_bin_edges_CPC3 = np.linspace(all_ptemps_CPC3.min(), all_ptemps_CPC3.max(), num_bins_ptemp + 1)
 
CPC3_bin_medians, _, _, _ = binned_statistic_2d(all_latitudes_CPC3, all_ptemps_CPC3, 
        all_CPC3_concs, statistic="mean", bins=[lat_bin_edges_CPC3, ptemp_bin_edges_CPC3])
 
##--Binning for CPC10 data--##
all_latitudes_CPC10 = np.concatenate([df["Latitude"].values for df in CPC10_dfs])
all_ptemps_CPC10 = np.concatenate([df["Ptemp"].values for df in CPC10_dfs])
all_CPC10_concs = np.concatenate([df["CPC10_conc"].values for df in CPC10_dfs])
 
lat_bin_edges_CPC10 = np.linspace(all_latitudes_CPC10.min(), all_latitudes_CPC10.max(), num_bins_lat + 1)
ptemp_bin_edges_CPC10 = np.linspace(all_ptemps_CPC10.min(), all_ptemps_CPC10.max(), num_bins_ptemp + 1)
 
CPC10_bin_medians, _, _, _ = binned_statistic_2d(all_latitudes_CPC10, all_ptemps_CPC10, 
        all_CPC10_concs, statistic="mean", bins=[lat_bin_edges_CPC10, ptemp_bin_edges_CPC10])
 
##--Binning for nucleating particle data--##
all_latitudes_nuc = np.concatenate([df["Latitude"].values for df in nuc_dfs])
all_ptemps_nuc = np.concatenate([df["Ptemp"].values for df in nuc_dfs])
all_nuc_particles = np.concatenate([df["nuc_particles"].values for df in nuc_dfs])
 
lat_bin_edges_nuc = np.linspace(all_latitudes_nuc.min(), all_latitudes_nuc.max(), num_bins_lat + 1)
ptemp_bin_edges_nuc = np.linspace(all_ptemps_nuc.min(), all_ptemps_nuc.max(), num_bins_ptemp + 1)
 
nuc_bin_medians, _, _, _ = binned_statistic_2d(all_latitudes_nuc, all_ptemps_nuc, 
        all_nuc_particles, statistic="mean", bins=[lat_bin_edges_nuc, ptemp_bin_edges_nuc])

##--Binning for growth N(10-89) particle data--##
all_latitudes_grow = np.concatenate([df["Latitude"].values for df in grow_dfs])
all_ptemps_grow = np.concatenate([df["Ptemp"].values for df in grow_dfs])
all_grow_particles = np.concatenate([df["n_10_89"].values for df in grow_dfs])
 
lat_bin_edges_grow = np.linspace(all_latitudes_grow.min(), all_latitudes_grow.max(), num_bins_lat + 1)
ptemp_bin_edges_grow = np.linspace(all_ptemps_grow.min(), all_ptemps_grow.max(), num_bins_ptemp + 1)
 
grow_bin_medians, _, _, _ = binned_statistic_2d(all_latitudes_grow, all_ptemps_grow, 
    all_grow_particles, statistic="mean", bins=[lat_bin_edges_grow, ptemp_bin_edges_grow])
 
################
##--PLOTTING--##
################
 
def plot_curtain(bin_medians, x_edges, y_edges, vmin, vmax, title, cbar_label, output_path):
    fig, ax = plt.subplots(figsize=(8, 6))
 
    ##--Makecolor map where 0 values are white--##
    new_cmap = plt.get_cmap('viridis')
    new_cmap.set_under('w')
 
    ##--Plot the 2D data using pcolormesh--##
    mesh = ax.pcolormesh(x_edges, y_edges, bin_medians.T, shading="auto", cmap=new_cmap, vmin=vmin, vmax=vmax)
 
    ##--Add colorbar--##
    cb = fig.colorbar(mesh, ax=ax)
    cb.minorticks_on()
    cb.ax.tick_params(labelsize=16)
    cb.set_label(cbar_label, fontsize=16)
    
    ##--Add dashed horizontal lines for the polar dome boundaries--##
    ##--Boundaries are defined from Bozem et al 2019 (ACP)--##
    ax.axhline(y=285, color='k', linestyle='--', linewidth=1)
    ax.axhline(y=299, color='k', linestyle='--', linewidth=1)
    
    '''
    ##--Add text labels on the left-hand side within the plot area--##
    ##--Compute midpoints for label placement--##
    polar_dome_mid = (238 + 275) / 2
    marginal_polar_dome_mid = (275 + 299) / 2
    x_text = ax.get_xlim()[0] - 0.25  # left edge plus a small offset
    
    ax.text(x_text, polar_dome_mid, 'Polar Dome',
            rotation=90, fontsize=10, color='k',
            verticalalignment='center', horizontalalignment='center')
    ax.text(x_text, marginal_polar_dome_mid, 'Marginal Polar Dome',
            rotation=90, fontsize=10, color='k',
            verticalalignment='center', horizontalalignment='center')
    '''
    ##--Set axis labels and title--##
    ax.set_xlabel("Latitude (°)", fontsize=16)
    ax.set_ylabel("Potential Temperature \u0398 (K)", fontsize=16)
    ax.tick_params(axis='both', labelsize=16)
    ax.set_title(title, fontsize=18)
    #ax.set_ylim(238, 301)
    #ax.set_xlim(79.5, 83.7)
 
    ##--Save the plot--##
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.tight_layout()
    plt.show()
 
##--Plot for rBC--##
plot_curtain(BC_bin_medians, lat_bin_edges_BC, ptemp_bin_edges_BC, vmin=1, vmax=150,
    title="rBC Particle Abundance", cbar_label="rBC Particles $(Counts/cm^{3})$",
    output_path=f"{output_path}\\BC/PTempLatitude/MultiFlights.png")

##--Plot for CPC3--##
plot_curtain(CPC3_bin_medians, lat_bin_edges_CPC3, ptemp_bin_edges_CPC3, vmin=1, vmax=2000,
    title="Particles >2.5 nm Abundance", cbar_label="Particles >2.5 nm $(Counts/cm^{3})$",
    output_path=f"{output_path}\\CPC3/PTempLatitude/MultiFlights.png")

##--Plot for CPC10--##
plot_curtain(CPC10_bin_medians, lat_bin_edges_CPC10, ptemp_bin_edges_CPC10, vmin=1, vmax=1400,
    title="Particles >10 nm Abundance", cbar_label="Particles >10 nm $(Counts/cm^{3})$",
    output_path=f"{output_path}\\CPC10/PTempLatitude/MultiFlights.png")
 
##--Plot for nucleating particles--##
plot_curtain(nuc_bin_medians, lat_bin_edges_nuc, ptemp_bin_edges_nuc, vmin=1, vmax=1000,
    title="2.5-10 nm Particle Abundance", cbar_label="2.5-10 nm Particles $(Counts/cm^{3})$",
    output_path=f"{output_path}\\Nucleating/PTempLatitude/MultiFlights.png")

##--Plot for N(10-89)--##
plot_curtain(grow_bin_medians, lat_bin_edges_grow, ptemp_bin_edges_grow, vmin=0, vmax=1000,
    title="10-89 nm Particle Abundance", cbar_label="10-89 nm Particles $(Counts/cm^{3})$",
    output_path=f"{output_path}\\N_10_89/PTempLatitude/MultiFlights.png")


########################
##--Diagnostic Plots--##
########################

##--Remove hashtags below to comment out this section--##
'''

##--Counts per bin for CPC3 data--##
CPC3_bin_counts, _, _, _ = binned_statistic_2d(all_latitudes_CPC3, all_ptemps_CPC3, all_CPC3_concs,
    statistic="count", bins=[lat_bin_edges_CPC3, ptemp_bin_edges_CPC3])
 
##--Counts per bin for CPC10 data--##
CPC10_bin_counts, _, _, _ = binned_statistic_2d(all_latitudes_CPC10, all_ptemps_CPC10, all_CPC10_concs,
    statistic="count", bins=[lat_bin_edges_CPC10, ptemp_bin_edges_CPC10])
 
##--Counts per bin for N3-10 particles--##
nuc_bin_counts, _, _, _ = binned_statistic_2d(all_latitudes_nuc, all_ptemps_nuc, all_nuc_particles,
    statistic="count", bins=[lat_bin_edges_nuc, ptemp_bin_edges_nuc])

##--Counts per bin for N10-89 particles--##
grow_bin_counts, _, _, _ = binned_statistic_2d(all_latitudes_grow, all_ptemps_grow, all_grow_particles,
    statistic="count", bins=[lat_bin_edges_grow, ptemp_bin_edges_grow])

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
    cb.ax.tick_params(labelsize=16)
    cb.set_label(cbar_label, fontsize=16)
    
    ##--Add dashed horizontal lines for the polar dome boundaries--##
    ax.axhline(y=285, color='k', linestyle='--', linewidth=1)
    ax.axhline(y=299, color='k', linestyle='--', linewidth=1)
    

    ##--Add labels on the left-hand side within the plot area--##
    polar_dome_mid = (238 + 275) / 2
    marginal_polar_dome_mid = (275 + 299) / 2
    x_text = ax.get_xlim()[0] - 0.25 
    
    ax.text(x_text, polar_dome_mid, 'Polar Dome',
            rotation=90, fontsize=10, color='k',
            verticalalignment='center', horizontalalignment='center')
    ax.text(x_text, marginal_polar_dome_mid, 'Marginal Polar Dome',
            rotation=90, fontsize=10, color='k',
            verticalalignment='center', horizontalalignment='center')

 
    ##--Set axis labels and title--##
    ax.set_xlabel("Latitude (°)", fontsize=16)
    ax.set_ylabel("Potential Temperature Θ (K)", fontsize=16)
    ax.tick_params(axis='both', labelsize=16)
    ax.set_title(title, fontsize=18)
    #ax.set_ylim(238, 301)
    #ax.set_xlim(79.5, 83.7)
 
    ##--Save the plot--##
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.tight_layout()
    plt.show()
 
##--Plot for CPC3 counts--##
plot_curtain(CPC3_bin_counts, lat_bin_edges_CPC3, ptemp_bin_edges_CPC3, vmin=1, vmax=5500, 
    title="Particles >2.5 nm Data Point Counts", cbar_label="Number of Data Points",
    output_path=f"{output_path}\\CPC3/PtempLatitude/MultiFlights_diagnostic.png")
 
##--Plot for CPC10 counts--##
plot_curtain(CPC10_bin_counts, lat_bin_edges_CPC10, ptemp_bin_edges_CPC10, vmin=1, vmax=6000,  
    title="Particles >10 nm Data Point Counts", cbar_label="Number of Data Points",
    output_path=f"{output_path}\\CPC10/PtempLatitude/MultiFlights_diagnostic.png")
 
##--Plot for N3-10 counts--##
plot_curtain(nuc_bin_counts, lat_bin_edges_nuc, ptemp_bin_edges_nuc, vmin=1, vmax=4000,  
    title="2.5-10 nm Data Point Counts", cbar_label="Number of Data Points",
    output_path=f"{output_path}\\Nucleating/PtempLatitude/MultiFlights_diagnostic.png")

##--Plot for N10-89 counts--##
plot_curtain(grow_bin_counts, lat_bin_edges_grow, ptemp_bin_edges_grow, vmin=1, vmax=3500,  
    title="10-89 nm Data Point Counts", cbar_label="Number of Data Points",
    output_path=f"{output_path}\\N_10_89/PtempLatitude/MultiFlights_diagnostic.png")
'''
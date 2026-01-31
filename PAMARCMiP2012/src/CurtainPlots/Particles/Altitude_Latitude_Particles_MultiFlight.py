# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 15:40:00 2026

@author: repooley
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from scipy.stats import binned_statistic_2d
from datetime import date
 
###################
##--User inputs--##
###################

##--Set the base directory to project folder--##
directory = r"C:\Users\repooley\REP_PhD\Arctic_NPF\PAMARCMiP2012\data\raw"
 
##--Choose which flights to analyze here!--##
flights_to_analyze = ["Flight1", "Flight2", "Flight3", "Flight6", "Flight7", "Flight8", "Flight9"]

##--Set binning for PTemp and Latitude--##
num_bins_lat = 8
num_bins_alt = 8

##--Bin data are in a CSV file--##
##--SAME bins for NETCARE and PAMARCMiP--##
UHSAS_bins = pd.read_csv(r"C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data\raw\NETCARE2015_UHSAS_bins.csv")

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
def find_files(directory, flight, partial_name):
    search_pattern = os.path.join(directory, flight, f"*{partial_name}*")
    return sorted(glob.glob(search_pattern))
 
##--Store processed data here: --##
CPC10_dfs = []
grow_dfs = []
total_dfs = []
 
##--Loop through each flight, pulling and analyzing data--##
for flight in flights_to_analyze:
    ##--Follow which flight is processing--##
    print(f"Processing {flight}...")
    
    ##--Pull file--##
    data = pd.read_csv(find_files(directory, flight, ".csv")[0])
    
    #################
    ##--Pull data--##
    #################
    
    ##--Data--##
    altitude = data['Altitude'] # in m
    latitude = data['Latitude'] # in degrees
    temperature = data['Temp'] + 273.15 # in K
    pressure = data['Pressure'] # in pa
    time = data['Time'] # seconds since midnight
    
    ##--The first datapoint in 'latitude' column is erraneous (47.12 N)--##
    latitude = latitude.where(latitude >= 50, np.nan)
    
    ##--USHAS Data--##
    UHSAS_total_num = data['UH-TotConc'] # particles/cm^3
    
    ##--Bin data are in a CSV file--##
    ##--SAME bins for NETCARE and PAMARCMiP--##
    UHSAS_bins = pd.read_csv(r"C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data\raw\NETCARE2015_UHSAS_bins.csv")
    
    ##--Make list of columns to pull, each named bin_x--##
    UHSAS_bin_num = [f'UH_bin_{i}' for i in range(1, 100)]
    
    ##--Information for bins 1 thru 99--##
    UHSAS_bin_center = UHSAS_bins['bin_avg'].iloc[0:100]
    UHSAS_lower_bound = UHSAS_bins['lower_bound'].iloc[0:100]
    UHSAS_upper_bound = UHSAS_bins['upper_bound'].iloc[0:100]
    
    ##--Put column names and content in a dictionary and then convert to a Pandas df--##
    UHSAS_bins = pd.DataFrame({col: data[col] for col in UHSAS_bin_num})
    
    ##--Create new column names by rounding the bin center values to the nearest integer--##
    UHSAS_new_col_names = UHSAS_bin_center.round().astype(int).tolist()
    
    ##--Rename the UHSAS_bins df columns to bin average values--##
    UHSAS_bins.columns = UHSAS_new_col_names
    
    ##--10 nm CPC data--##
    CPC10_conc = data['CPC10'] # count/cm^3
    
    ##--REMOVE OUTLIERS above 99th percentile--##
    p = 0.99
    
    ##--Compute threshold for each UHSAS column--##
    uhsas_thresh = UHSAS_bins.quantile(p)
    
    ##--keep only rows where each bin is below its threshold--##
    UHSAS_bins_filtered = UHSAS_bins[UHSAS_bins.le(uhsas_thresh, axis=1)]
    
    cpc10_thresh = CPC10_conc.quantile(p)
    CPC10_filtered = CPC10_conc[CPC10_conc <= cpc10_thresh]
    
    ####################################
    ##--Assign date to flight number--##
    ####################################
    
    if flight=="Flight1":
        flight_date = date(2012, 3, 29)
    elif flight=="Flight2":
        flight_date = date(2012, 3, 30)
    elif flight=="Flight3":
        flight_date = date(2012, 4, 2)
    elif flight=="Flight4" or flight=="Flight5":
        flight_date = date(2012, 4, 3)
    elif flight=="Flight6":
        flight_date = date(2012, 4, 4)
    elif flight=="Flight7": 
        flight_date = date(2012, 4, 5)
    elif flight=="Flight8": 
        flight_date = date(2012, 4, 6)
    elif flight=="Flight9": 
        flight_date = date(2012, 4, 7)
    
    ###############################
    ##--De-Normalize UHSAS Data--##
    ###############################
    
    ##--For total count calculation--##
    
    ##--Calculate dlogDp for UHSAS bins--##
    UHSAS_dlogDp = np.log(UHSAS_upper_bound.values) - np.log(UHSAS_lower_bound.values)
    
    ##--Get only particle count data (excluding 'Time')--##
    UHSAS_particle_counts = UHSAS_bins.loc[:, UHSAS_new_col_names]  # Adjust column names as needed
    
    ##--De-Normalize counts by multiplying by dlogDp across all rows--##
    UHSAS_denorm_counts = UHSAS_particle_counts.multiply(UHSAS_dlogDp, axis=1)
    
    ##################
    ##--UHSAS bins--##
    ##################

    ##--Bin data are in a CSV file--##
    ##--SAME bins for NETCARE and PAMARCMiP--##
    UHSAS_bins = pd.read_csv(r"C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data\raw\NETCARE2015_UHSAS_bins.csv")
    
    ##--Make list of columns to pull, each named bin_x--##
    UHSAS_bin_num = [f'UH_bin_{i}' for i in range(1, 100)]
    
    ##--Information for bins 1 thru 99--##
    UHSAS_bin_center = UHSAS_bins['bin_avg'].iloc[0:100]
    UHSAS_lower_bound = UHSAS_bins['lower_bound'].iloc[0:100]
    UHSAS_upper_bound = UHSAS_bins['upper_bound'].iloc[0:100]
    
    ##--Put column names and content in a dictionary and then convert to a Pandas df--##
    UHSAS_bins = pd.DataFrame({col: data[col] for col in UHSAS_bin_num})
    
    ##--Create new column names by rounding the bin center values to the nearest integer--##
    UHSAS_new_col_names = UHSAS_bin_center.round().astype(int).tolist()
    
    ##--Rename the UHSAS_bins df columns to bin average values--##
    UHSAS_bins.columns = UHSAS_new_col_names
    
    ##--REMOVE OUTLIERS above 99th percentile--##
    p = 0.99
    
    ##--Compute threshold for each UHSAS column--##
    uhsas_thresh = UHSAS_bins.quantile(p)
    
    ##--keep only rows where each bin is below its threshold--##
    UHSAS_bins_filtered = UHSAS_bins[UHSAS_bins.le(uhsas_thresh, axis=1)]

    #####################
    ##--Calc N(10-60)--##
    #####################
    
    ##--Create df with UHSAS total counts--##
    UHSAS_total = pd.DataFrame({'Time': time, 'Total_count': UHSAS_total_num})
    
    ##--Create df with CPC10 counts and set index to time--##
    CPC10_counts = pd.DataFrame({'Time':time, 'Counts':CPC10_filtered})
    
    ##--Calculate particles below UHSAS lower cutoff--##
    n_10_60 = (CPC10_counts['Counts'] - UHSAS_total['Total_count'])
    
    ##--Change calculated particle counts less than zero to NaN--##
    n_10_60 = np.where(n_10_60 >= 0, n_10_60, np.nan)
    
    ##--Put N(10-60) bin center in a df--##
    n_10_60_center = pd.DataFrame([35])
    
    ##--Flatten--##
    n_10_60_center = pd.Series(n_10_60_center.values.flatten())
    
    ##--Convert n_10_60 to a df--##
    n_10_60_df = pd.DataFrame({'35': n_10_60, 'time':time})
            
    #########################
    ##--Create dataframes--##
    #########################
    
    ##--Place all binned data in a single df--##
    all_bins_aligned = pd.concat([n_10_60_df, UHSAS_bins_filtered], axis=1)
    
    ##--Set time as the index--##
    #ll_bins_aligned = all_bins_aligned.set_index('time')
    
    total_particle_count = all_bins_aligned.drop(columns=['time']).sum(axis=1, numeric_only=True)

    ##--Drop NaNs, done for individual datasets for data preservation--##
    CPC10_df = pd.DataFrame({'Altitude': altitude, 'Latitude': latitude, 
                            'CPC10_conc': CPC10_filtered}).dropna()
    ##--Calling n 10-60 'growth'--##
    grow_df = pd.DataFrame({'Altitude': altitude, 'Latitude': latitude, 
                           'n_10_60': n_10_60}).dropna()
    
    ##--Total counts--##
    total_df = pd.DataFrame({'Altitude': altitude, 'Latitude': latitude, 
                             'total_count': total_particle_count}).dropna()

    ##--Store all processed data and ensure in numpy arrays--##
    CPC10_dfs.append(CPC10_df[['Altitude', 'Latitude', 'CPC10_conc']])
    grow_dfs.append(grow_df[['Altitude', 'Latitude', 'n_10_60']])
    total_dfs.append(total_df[['Altitude', 'Latitude', 'total_count']])

###########################
##--Prepare for Binning--##
###########################

##--Binning for CPC10 data--##
all_latitudes_CPC10 = np.concatenate([df["Latitude"].values for df in CPC10_dfs])
all_altitudes_CPC10 = np.concatenate([df["Altitude"].values for df in CPC10_dfs])
all_CPC10_concs = np.concatenate([df["CPC10_conc"].values for df in CPC10_dfs])
 
lat_bin_edges_CPC10 = np.linspace(all_latitudes_CPC10.min(), all_latitudes_CPC10.max(), num_bins_lat + 1)
alt_bin_edges_CPC10 = np.linspace(all_altitudes_CPC10.min(), all_altitudes_CPC10.max(), num_bins_alt + 1)
 
CPC10_bin_medians, _, _, _ = binned_statistic_2d(all_latitudes_CPC10, all_altitudes_CPC10, 
    all_CPC10_concs, statistic="median", bins=[lat_bin_edges_CPC10, alt_bin_edges_CPC10])
 
##--Binning for growth N(10-60) particle data--##
all_latitudes_grow = np.concatenate([df["Latitude"].values for df in grow_dfs])
all_altitudes_grow = np.concatenate([df["Altitude"].values for df in grow_dfs])
all_grow_particles = np.concatenate([df["n_10_60"].values for df in grow_dfs])
 
lat_bin_edges_grow = np.linspace(all_latitudes_grow.min(), all_latitudes_grow.max(), num_bins_lat + 1)
alt_bin_edges_grow = np.linspace(all_altitudes_grow.min(), all_altitudes_grow.max(), num_bins_alt + 1)
 
grow_bin_medians, _, _, _ = binned_statistic_2d(all_latitudes_grow, all_altitudes_grow, 
    all_grow_particles, statistic="median", bins=[lat_bin_edges_grow, alt_bin_edges_grow])

##--Binning for total count particle data--##
all_latitudes_total = np.concatenate([df["Latitude"].values for df in total_dfs])
all_altitudes_total = np.concatenate([df["Altitude"].values for df in total_dfs])
all_total_particles = np.concatenate([df["total_count"].values for df in total_dfs])
 
lat_bin_edges_total = np.linspace(all_latitudes_total.min(), all_latitudes_total.max(), num_bins_lat + 1)
alt_bin_edges_total = np.linspace(all_altitudes_total.min(), all_altitudes_total.max(), num_bins_alt + 1)
 
total_bin_medians, _, _, _ = binned_statistic_2d(all_latitudes_total, all_altitudes_total, 
    all_total_particles, statistic="median", bins=[lat_bin_edges_total, alt_bin_edges_total])
 
################
##--PLOTTING--##
################
 
def plot_curtain(bin_medians, x_edges, y_edges, vmin, vmax, title, cbar_label):
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
    #plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.tight_layout()
    plt.show()
 
##--Plot for CPC10--##
plot_curtain(CPC10_bin_medians, lat_bin_edges_CPC10, alt_bin_edges_CPC10, vmin=0, vmax=1000,
    title="Particles >10 nm Abundance", cbar_label="Particles >10 nm $(Counts/cm^{3})$")
    #output_path=f"{output_path}\\CPC10/AltitudeLatitude/MultiFlights.png")

##--Plot for N(10-60)--##
plot_curtain(grow_bin_medians, lat_bin_edges_grow, alt_bin_edges_grow, vmin=0, vmax=1000,
    title="10-89 nm Particle Abundance", cbar_label="10-89 nm Particles $(Counts/cm^{3})$")
    #output_path=f"{output_path}\\N_10_89/AltitudeLatitude/MultiFlights.png")
    
##--Plot for total count--##
plot_curtain(total_bin_medians, lat_bin_edges_total, alt_bin_edges_total, vmin=0, vmax=1000,
    title="Total Particle Abundance", cbar_label="Total Particles $(Counts/cm^{3})$")
    #output_path=f"{output_path}\\N_10_89/AltitudeLatitude/MultiFlights.png")

########################
##--Diagnostic Plots--##
########################

##--Remove hashtags below to comment out this section--##
#'''
 
##--Counts per bin for CPC10 data--##
CPC10_bin_counts, _, _, _ = binned_statistic_2d(all_latitudes_CPC10, all_altitudes_CPC10, all_CPC10_concs,
    statistic="count", bins=[lat_bin_edges_CPC10, alt_bin_edges_CPC10])
 

##--Counts per bin for N10-89 particles--##
grow_bin_counts, _, _, _ = binned_statistic_2d(all_latitudes_grow, all_altitudes_grow, all_grow_particles,
    statistic="count", bins=[lat_bin_edges_grow, alt_bin_edges_grow])

##--Plotting--##

def plot_curtain(bin_counts, x_edges, y_edges, vmin, vmax, title, cbar_label):
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
    #plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.tight_layout()
    plt.show()
 
##--Plot for CPC10 counts--##
plot_curtain(CPC10_bin_counts, lat_bin_edges_CPC10, alt_bin_edges_CPC10, vmin=1, vmax=7500,  
    title="Particles >10 nm Data Point Counts", cbar_label="Number of Data Points")

##--Plot for N10-89 counts--##
plot_curtain(grow_bin_counts, lat_bin_edges_grow, alt_bin_edges_grow, vmin=1, vmax=3500,  
    title="10-89 nm Data Point Counts", cbar_label="Number of Data Points")

#'''
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 10:02:14 2026

@author: repooley
"""

import icartt
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from scipy.stats import binned_statistic_2d
from datetime import date
import matplotlib.ticker as ticker

###################
##--User inputs--##
###################

##--Set the base directory to project folder--##
directory = r"C:\Users\repooley\REP_PhD\Arctic_NPF\ATom2018\data"

##--The flight10 file also includes flight11--##
flights_to_analyze = ["Flight2", "Flight10", "Flight11", "Flight12"]

##--Set binning for PTemp and Latitude--##
num_bins_lat = 6
num_bins_ptemp = 6

##--Define function that creates datasets from filenames--##
def find_files(directory, flight):
    ##--flight data are stored in a folder called "raw"--##
    flight_dir = os.path.join(directory, "raw", flight)
    search_pattern = os.path.join(flight_dir, "*.ict")
    return sorted(glob.glob(search_pattern))

nuc_dfs = []
grow_dfs = []
rBC_dfs = []

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
    nucleating = dataset.data['N_nucl_AMP'] # num/cm^3 STP (2.7-12 nm)
    aitken = dataset.data['N_aitken_AMP'] # num/cm^3 STP (12-60 nm)
    latitude = dataset.data['LAT_AMSSD'] # deg
    rBC = dataset.data['BC_mass_90_550_nm'] # ng/m^3 (std)
    
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

    ###########################
    ##--Place in dataframes--##
    ###########################
    
    ##--2.5-10nm, 'nucleating'--##
    nuc_df = pd.DataFrame({'PTemp': potential_temp, 'Latitude': latitude, 'nuc_particles': nucleating}).dropna()
    ##--Drop rows where the potential temperature is above 310 K for comparison to other campaigns--##
    nuc_df = nuc_df[nuc_df['PTemp']<310]
    ##--And latitudes below 66.5 degrees (below the Arctic circle)--##
    nuc_df = nuc_df[nuc_df['Latitude']>66.5]
    
    nuc_dfs.append(nuc_df)
    
    ##--10-89nm, 'growth'--##
    grow_df = pd.DataFrame({'PTemp': potential_temp, 'Latitude': latitude, 'grow_particles': aitken}).dropna()
    ##--Drop rows where the potential temperature is above 310 K for comparison to other campaigns--##
    grow_df = grow_df[grow_df['PTemp']<310]
    ##--And latitudes below 66.5 degrees (below the Arctic circle)--##
    grow_df = grow_df[grow_df['Latitude']>66.5]
    
    grow_dfs.append(grow_df)
    
    ##--rBC--##
    rBC_df = pd.DataFrame({'PTemp': potential_temp, 'Latitude': latitude, 'rBC': rBC}).dropna()
    ##--Drop rows where the potential temperature is above 310 K for comparison to other campaigns--##
    rBC_df = rBC_df[rBC_df['PTemp']<310]
    ##--And latitudes below 66.5 degrees (below the Arctic circle)--##
    rBC_df = rBC_df[rBC_df['Latitude']>66.5]
    
    rBC_dfs.append(rBC_df)
    
###########################
##--Create 2D histogram--##
###########################

##--Binning for nucleating particle data--##
all_latitudes_nuc = np.concatenate([df["Latitude"].values for df in nuc_dfs])
all_ptemps_nuc = np.concatenate([df["PTemp"].values for df in nuc_dfs])
all_nuc_particles = np.concatenate([df["nuc_particles"].values for df in nuc_dfs])
 
lat_bin_edges_nuc = np.linspace(all_latitudes_nuc.min(), all_latitudes_nuc.max(), num_bins_lat + 1)
ptemp_bin_edges_nuc = np.linspace(all_ptemps_nuc.min(), all_ptemps_nuc.max(), num_bins_ptemp + 1)
 
nuc_bin_medians, _, _, _ = binned_statistic_2d(all_latitudes_nuc, all_ptemps_nuc, 
        all_nuc_particles, statistic="median", bins=[lat_bin_edges_nuc, ptemp_bin_edges_nuc])

##--Binning for growth N(12-60) particle data--##
all_latitudes_grow = np.concatenate([df["Latitude"].values for df in grow_dfs])
all_ptemps_grow = np.concatenate([df["PTemp"].values for df in grow_dfs])
all_grow_particles = np.concatenate([df["grow_particles"].values for df in grow_dfs])
 
lat_bin_edges_grow = np.linspace(all_latitudes_grow.min(), all_latitudes_grow.max(), num_bins_lat + 1)
ptemp_bin_edges_grow = np.linspace(all_ptemps_grow.min(), all_ptemps_grow.max(), num_bins_ptemp + 1)
 
grow_bin_medians, _, _, _ = binned_statistic_2d(all_latitudes_grow, all_ptemps_grow, 
    all_grow_particles, statistic="median", bins=[lat_bin_edges_grow, ptemp_bin_edges_grow])

##--Binning for rBC data--##
all_latitudes_rBC = np.concatenate([df["Latitude"].values for df in rBC_dfs])
all_ptemps_rBC = np.concatenate([df["PTemp"].values for df in rBC_dfs])
all_rBC_counts = np.concatenate([df["rBC"].values for df in rBC_dfs])
 
lat_bin_edges_rBC = np.linspace(all_latitudes_rBC.min(), all_latitudes_rBC.max(), num_bins_lat + 1)
ptemp_bin_edges_rBC = np.linspace(all_ptemps_rBC.min(), all_ptemps_rBC.max(), num_bins_ptemp + 1)
 
rBC_bin_medians, _, _, _ = binned_statistic_2d(all_latitudes_rBC, all_ptemps_rBC, 
        all_rBC_counts, statistic="median", bins=[lat_bin_edges_rBC, ptemp_bin_edges_rBC])

################
##--PLOTTING--##
################
 
def plot_curtain(bin_medians, x_edges, y_edges, vmin, vmax, title, cbar_label):
    fig, ax = plt.subplots(figsize=(6, 6))
 
    ##--Makecolor map where 0 values are white--##
    new_cmap = plt.get_cmap('viridis')
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
    ax.set_ylabel("Potential Temperature \u0398 (K)", fontsize=18)
    ax.tick_params(axis='both', labelsize=18)
    ax.set_title(title, fontsize=20)
    #ax.set_ylim(238, 316)
    #ax.set_xlim(64, 86)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))

    plt.tight_layout()
    plt.show()
 
##--Plot for nucleating particles--##
plot_curtain(nuc_bin_medians, lat_bin_edges_nuc, ptemp_bin_edges_nuc, vmin=1, vmax=1000,
    title="2.5-10 nm Particle Abundance", cbar_label="2.5-10 nm Particles $(Counts/cm^{3})$")

##--Plot for N(12-60)--##
plot_curtain(grow_bin_medians, lat_bin_edges_grow, ptemp_bin_edges_grow, vmin=0, vmax=1000,
    title="10-89 nm Particle Abundance", cbar_label="10-89 nm Particles $(Counts/cm^{3})$")

##--Plot for rBC--##
plot_curtain(rBC_bin_medians, lat_bin_edges_rBC, ptemp_bin_edges_rBC, vmin=0, vmax=75,
    title="rBC Particle Abundance", cbar_label="rBC Particles $(ng/m^{3})$")
 
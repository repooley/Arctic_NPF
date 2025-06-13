# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 09:20:50 2025

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
 
##--Define number of bins here--##
num_bins_lat = 10
num_bins_ptemp = 10

# add base output path

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
 
##--Choose which flights to analyze here!--##
flights_to_analyze = ["Flight2", "Flight3", "Flight4", "Flight5", "Flight6", "Flight7", "Flight8", "Flight9", "Flight10"]
 
##--Store processed data here: --##
RH_w_dfs = []
RH_i_dfs = []
 
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
 
    ##--Pull H2O files--##
    H2O_files = find_files(flight_dir, 'H2O')

    if H2O_files:
        H2O = icartt.Dataset(H2O_files[0])
    else:
        print(f"Missing H2O data for {flight}. Skipping...")
        continue
 
    #########################
    ##--Pull & align data--##
    #########################
    
    ##--AIMMS Data--##
    altitude = aimms.data['Alt'] # in m
    latitude = aimms.data['Lat'] # in degrees
    temperature = aimms.data['Temp'] 
    pressure = aimms.data['BP'] # in pa
    aimms_time =aimms.data['TimeWave'] # seconds since midnight
    
    ##--H2O data--##
    H2O_time = H2O.data['Time_UTC'] # seconds since midnight
    H2O_conc = H2O.data['H2O_ppmv'] # ppmv
    
    H2O_df = pd.DataFrame({'time': H2O_time, 'conc': H2O_conc}).set_index('time')
    H2O_conc_aligned = H2O_df.reindex(aimms_time)['conc']

    ####################
    ##--Calculations--##
    ####################

    ##--Convert H2O ppm to RH wrt Water--##

    ##--Lowe and Ficke (1974) 6th deg polynomial approach--##
    ##--Sat vap pressure water -50 to 50 C--##
    wa0 = 6.107799961
    wa1 = 4.436518521E-1
    wa2 = 1.428945805E-2
    wa3 = 2.650648471E-4
    wa4 = 3.031240396E-6
    wa5 = 2.034080948E-8
    wa6 = 6.136820929E-11

    ##--Generate empty lists for RH wrt water outputs--##
    saturation_humidity_w = []
    relative_humidity_w = []

    ##--Calculate saturation humidity in ppmv and relative humidity--##
    for T, P, H2O_ppmv in zip(temperature, pressure, H2O_conc_aligned):
        ##--Only calculate within temp range--##
        if -50 <= T < 50:
            ##--saturation vapor pressure using Lowe and Ficke (1974) eqn--##
            e_sw = wa0 + wa1*T + wa2*(T**2)+ wa3*(T**3)+ wa4*(T**4) + wa5*(T**5) + wa6*(T**6) # in mbar 
            ##--Convert from mbar to pa--##
            e_sw_pa = e_sw*100
            ##--Saturation mixing ratio in ppmv--##
            w_s_ppmv = (e_sw_pa / P) * 1e6
            saturation_humidity_w.append(w_s_ppmv)
            ##--Relative humidity--##
            RH = (H2O_ppmv / w_s_ppmv) * 100  # in %
            relative_humidity_w.append(RH)
        else:
            saturation_humidity_w.append(np.nan)  
            relative_humidity_w.append(np.nan)    

    ##--With respect to ice--##

    ##--Lowe and Ficke (1974) 6th deg polynomial approach--##
    ##--Sat vap pressure ice -50 to 0 C--##
    ia0 = 6.109177956
    ia1 = 5.034698970E-1
    ia2 = 1.886013408E-2
    ia3 = 4.176223716E-4
    ia4 = 5.824720280E-6
    ia5 = 4.838803174E-8
    ia6 = 1.838826904E-10

    ##--Generate empty lists for humidity outputs--##
    saturation_humidity_i = []
    relative_humidity_i = []

    ##--Calculate saturation humidity wrt ice in ppmv and RH--##
    for T, P, H2O_ppmv in zip(temperature, pressure, H2O_conc_aligned):
        ##--Only calculate within temp range--##
        if -50 <= T < 0:
            ##--Saturation vapor pressure using Lowe and Ficke (1974) eqn--##
            e_si = ia0 + ia1*T + ia2*(T**2) + ia3*(T**3) + ia4*(T**4) + ia5*(T**5) + ia6*(T**6)  # in mbar
            ##--Convert from mbar to Pa--##
            e_si_pa = e_si * 100
            ##--Saturation mixing ratio in ppbv--##
            e_si_ppmv = (e_si_pa / P) * 1e6
            saturation_humidity_i.append(e_si_ppmv)
            ##--Relative Humidity--##
            RH_i = (H2O_ppmv / e_si_ppmv) * 100  # in %
            relative_humidity_i.append(RH_i)
        else:
            saturation_humidity_i.append(np.nan)  
            relative_humidity_i.append(np.nan)    

    ##--Convert absolute temperature to potential temperature--##
    ##--Constants--##
    p_0 = 1E5 # Reference pressure in Pa (1000 hPa)
    k = 0.286 # Poisson constant for dry air

    ##--Convert temperature from Celcius to Kelvin--##
    temperature_k = np.array(temperature) + 273.15

    ##--Generate empty list for potential temperature output--##
    potential_temp = []

    ##--Calculate potential temperature from ambient temp & pressure--##
    for T, P in zip(temperature_k, pressure):
        p_t = T*(p_0/P)**k
        potential_temp.append(p_t)

    #########################
    ##--Create dataframes--##
    #########################
    
    RH_w_df = pd.DataFrame({'Ptemp': potential_temp, 'Latitude': latitude, 
                            'Relative_Humidity_w': relative_humidity_w}).dropna()
    RH_i_df = pd.DataFrame({'Ptemp': potential_temp, 'Latitude': latitude, 
                            'Relative_Humidity_i': relative_humidity_i}).dropna()

    ##--Store all processed data and ensure in numpy arrays--##
    RH_w_dfs.append(RH_w_df[['Ptemp', 'Latitude', 'Relative_Humidity_w']])
    RH_i_dfs.append(RH_i_df[['Ptemp', 'Latitude', 'Relative_Humidity_i']])

###########################
##--Prepare for Binning--##
###########################
 
##--Binning for RH wrt water data--##
all_latitudes_RH_w = np.concatenate([df["Latitude"].values for df in RH_w_dfs])
all_ptemps_RH_w = np.concatenate([df["Ptemp"].values for df in RH_w_dfs])
all_RH_w = np.concatenate([df["Relative_Humidity_w"].values for df in RH_w_dfs])
 
lat_bin_edges_RH_w = np.linspace(all_latitudes_RH_w.min(), all_latitudes_RH_w.max(), num_bins_lat + 1)
ptemp_bin_edges_RH_w = np.linspace(all_ptemps_RH_w.min(), all_ptemps_RH_w.max(), num_bins_ptemp + 1)
 
RH_w_bin_medians, _, _, _ = binned_statistic_2d(all_latitudes_RH_w, all_ptemps_RH_w, 
        all_RH_w, statistic="mean", bins=[lat_bin_edges_RH_w, ptemp_bin_edges_RH_w])
 
##--Binning for RH wrt ice data--##
all_latitudes_RH_i = np.concatenate([df["Latitude"].values for df in RH_i_dfs])
all_ptemps_RH_i = np.concatenate([df["Ptemp"].values for df in RH_i_dfs])
all_RH_i = np.concatenate([df["Relative_Humidity_i"].values for df in RH_i_dfs])
 
lat_bin_edges_RH_i = np.linspace(all_latitudes_RH_i.min(), all_latitudes_RH_i.max(), num_bins_lat + 1)
ptemp_bin_edges_RH_i = np.linspace(all_ptemps_RH_i.min(), all_ptemps_RH_i.max(), num_bins_ptemp + 1)
 
RH_i_bin_medians, _, _, _ = binned_statistic_2d(all_latitudes_RH_i, all_ptemps_RH_i, 
        all_RH_i, statistic="mean", bins=[lat_bin_edges_RH_i, ptemp_bin_edges_RH_i])
 
################
##--PLOTTING--##
################
 
def plot_curtain(bin_medians, x_edges, y_edges, vmin, vmax, title, cbar_label, output_path):
    fig, ax = plt.subplots(figsize=(8, 6))
 
    ##--Make special color map where 0 values are white--##
    new_cmap = plt.get_cmap('magma')
    new_cmap.set_under('w')
 
    ##--Plot the 2D data using pcolormesh--##
    mesh = ax.pcolormesh(x_edges, y_edges, bin_medians.T, shading="auto", cmap=new_cmap, vmin=vmin, vmax=vmax)
 
    ##--Add colorbar--##
    cb = fig.colorbar(mesh, ax=ax)
    cb.minorticks_on()
    cb.set_label(cbar_label, fontsize=12)
    
    ##--Add dashed horizontal lines for the polar dome boundaries--##
    ##--Boundaries are defined from Bozem et al 2019 (ACP)--##
    ax.axhline(y=285, color='k', linestyle='--', linewidth=1)
    ax.axhline(y=299, color='k', linestyle='--', linewidth=1)
    '''
    ##--Add text labels on the left-hand side within the plot area--##
    ##--Compute midpoints for label placement--##
    polar_dome_mid = (238 + 285) / 2
    marginal_polar_dome_mid = (285 + 299) / 2
    x_text = ax.get_xlim()[0] - 0.25  # left edge plus a small offset
    
    ax.text(x_text, polar_dome_mid, 'Polar Dome',
            rotation=90, fontsize=10, color='k',
            verticalalignment='center', horizontalalignment='center')
    ax.text(x_text, marginal_polar_dome_mid, 'Marginal Polar Dome',
            rotation=90, fontsize=10, color='k',
            verticalalignment='center', horizontalalignment='center')
    '''
    ##--Set axis labels and title--##
    ax.set_xlabel("Latitude (°)", fontsize=12)
    ax.set_ylabel("Potential Temperature \u0398 (K)", fontsize=12)
    ax.set_title(title)
    #ax.set_ylim(238, 301)
    #ax.set_xlim(79.5, 83.7)
 
    ##--Save the plot--##
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.tight_layout()
    plt.show()
 
##--Plot for RH wrt Water--##
plot_curtain(RH_w_bin_medians, lat_bin_edges_RH_w, ptemp_bin_edges_RH_w, vmin=0, vmax=120,
    title="Relative Humidity With Respect to Water", cbar_label="Percent Relative Humidity",
    output_path=r"C:\Users\repooley\REP_PhD\NETCARE2015\data\processed\CurtainPlots\Meteorological\PTempLatitude\RH_w_MultiFlights.png")

##--Plot for RH wrt Ice--##
plot_curtain(RH_i_bin_medians, lat_bin_edges_RH_i, ptemp_bin_edges_RH_i, vmin=0, vmax=120,
    title="Relative Humidity With Respect to Ice", cbar_label="Percent Relative Humidity",
    output_path=r"C:\Users\repooley\REP_PhD\NETCARE2015\data\processed\CurtainPlots\Meteorological\PTempLatitude\RH_i_MultiFlights.png")

########################
##--Diagnostic Plots--##
########################

##--Remove hashtags below to comment out this section--##
#'''

##--RH wrt water counts per bin data--##
RH_w_bin_counts, _, _, _ = binned_statistic_2d(all_latitudes_RH_w, all_ptemps_RH_w, all_RH_w,
    statistic="count", bins=[lat_bin_edges_RH_w, ptemp_bin_edges_RH_w])
 
##--RH wrt ice counts per bin data--##
RH_i_bin_counts, _, _, _ = binned_statistic_2d(all_latitudes_RH_i, all_ptemps_RH_i, all_RH_i,
    statistic="count", bins=[lat_bin_edges_RH_i, ptemp_bin_edges_RH_i])

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
    
    ##--Add dashed horizontal lines for the polar dome boundaries--##
    ax.axhline(y=285, color='k', linestyle='--', linewidth=1)
    ax.axhline(y=299, color='k', linestyle='--', linewidth=1)
    '''
    ##--Add labels on the left-hand side within the plot area--##
    polar_dome_mid = (248 + 285) / 2
    marginal_polar_dome_mid = (285 + 299) / 2
    x_text = ax.get_xlim()[0] - 0.25 
    
    ax.text(x_text, polar_dome_mid, 'Polar Dome',
            rotation=90, fontsize=10, color='k',
            verticalalignment='center', horizontalalignment='center')
    ax.text(x_text, marginal_polar_dome_mid, 'Marginal Polar Dome',
            rotation=90, fontsize=10, color='k',
            verticalalignment='center', horizontalalignment='center')
    '''
    ##--Set axis labels and title--##
    ax.set_xlabel("Latitude (°)", fontsize=12)
    ax.set_ylabel("Potential Temperature Θ (K)", fontsize=12)
    ax.set_title(title)
    #ax.set_ylim(238, 301)
    #ax.set_xlim(79.5, 83.7)
 
    ##--Save the plot--##
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.tight_layout()
    plt.show()
 
##--Plot for RH wrt water counts--##
plot_curtain(RH_w_bin_counts, lat_bin_edges_RH_w, ptemp_bin_edges_RH_w, vmin=1, vmax=3000, 
    title="RH wrt Water Data Point Counts", cbar_label="Number of Data Points",
    output_path=r"C:\Users\repooley\REP_PhD\NETCARE2015\data\processed\CurtainPlots\Meteorological\PTempLatitude\RH_w_MultiFlights_diagnostic.png")
 
##--Plot for RH wrt ice counts--##
plot_curtain(RH_w_bin_counts, lat_bin_edges_RH_w, ptemp_bin_edges_RH_w, vmin=1, vmax=3000,  
    title="RH wrt Ice Data Point Counts", cbar_label="Number of Data Points",
    output_path=r"C:\Users\repooley\REP_PhD\NETCARE2015\data\processed\CurtainPlots\Meteorological\PTempLatitude\RH_i_MultiFlights_diagnostic.png")

#'''
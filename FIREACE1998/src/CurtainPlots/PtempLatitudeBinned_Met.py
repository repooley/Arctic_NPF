# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 09:14:45 2025

@author: repooley
"""


import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
from scipy.stats import binned_statistic_2d
from datetime import date

###################
##--User inputs--##
###################

##--Set the base directory to project folder--##
directory = r"C:\Users\repooley\REP_PhD\Arctic_NPF\FIREACE1998\data\raw"

##--Flights to analyze - flights 1-18--##
flights_to_analyze = ["Flight3",  
                      "Flight7", "Flight8", "Flight9", "Flight10", "Flight11", "Flight12",
                      "Flight13", "Flight14", "Flight15", "Flight16", "Flight17", "Flight18"]

##--Set binning for PTemp and Latitude--##
##--Define number of bins here--##
num_bins_lat = 6
num_bins_ptemp = 12

##--Separate bin numbers for the averaged data--##
num_bins_lat_averaged = 6
num_bins_ptemp_averaged = 12

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

for flight in flights_to_analyze: 
    ##--'raw' contains a 1hz and 2min datafile, the 1hz one is always first--##
    data = pd.read_csv(find_files(directory, flight, "FIREACE")[0])
    
    ##--Pull data variables from file--##
    time = data['Time'] # HHMMSS UTC time
    pressure = data['Pressure'] * 100 # in Pa
    temperature = data['Temperature'] + 273.15 # in K
    RH_probe = data['RH'] # percent wrt water
    altitude = data['Altitude'] # in m (agl?)
    latitude = data['Latitude'] # degrees
    longitude = data['Longitude'] # degrees
    
    ##--Based on the supplied data, I strongly believe these two variables were swapped--##
    CO2_data = data['H2O'] # just labeled as 'mv' but there's clear pressure dependence
    H2O_data = data['CO2'] # 'mv'
    
    ####################################
    ##--Assign date to flight number--##
    ####################################
    
    if flight=="Flight1":
        flight_date = date(1998, 4, 8)
    elif flight=="Flight2":
        flight_date = date(1998, 4, 9)
    elif flight=="Flight3":
        flight_date = date(1998, 4, 12)
    elif flight=="Flight4":
        flight_date = date(1998, 4, 14)
    elif flight=="Flight5":
        flight_date = date(1998, 4, 15)
    elif flight=="Flight6":
        flight_date = date(1998, 4, 16)
    elif flight=="Flight7" or flight=="Flight8": 
        flight_date = date(1998, 4, 17)
    elif flight=="Flight9": 
        flight_date = date(1998, 4, 18)
    elif flight=="Flight10" or flight=="Flight11": 
        flight_date = date(1998, 4, 21)
    elif flight=="Flight12":
        flight_date = date(1998, 4, 22)
    elif flight=="Flight13":
        flight_date = date(1998, 4, 24)
    elif flight=="Flight14":
        flight_date = date(1998, 4, 25)
    elif flight=="Flight15":
        flight_date = date(1998, 4, 27)
    elif flight=="Flight16" or flight=="Flight17": 
        flight_date = date(1998, 4, 28)
    elif flight=="Flight18": 
        flight_date = date(1998, 4, 29)
    
    ###################
    ##--Conversions--##
    ###################
    
    ##--Convert to STP!--##
    ##--I believe the H2O and CO2 is, somehow, not in STP--##
    P_STP = 101325  # Pa
    T_STP = 273.15  # K
    
    ##--Create empty list for CO2--##
    CO2_STP = []
    
    for CO2, T, P in zip(CO2_data, temperature, pressure):
        if np.isnan(CO2) or np.isnan(T) or np.isnan(P):
            ##--Append with NaN if any input is NaN--##
            CO2_STP.append(np.nan)
        else:
            ##--Perform conversion if all inputs are valid--##
            CO2_conversion = CO2 * (P_STP / P) * (T / T_STP)
            CO2_STP.append(CO2_conversion)
        
    ##--Create empty list for H2O--##
    H2O_STP = []
    
    for H2O, T, P in zip(H2O_data, temperature, pressure):
        if np.isnan(H2O) or np.isnan(T) or np.isnan(P):
            ##--Append with NaN if any input is NaN--##
            H2O_STP.append(np.nan)
        else:
            ##--Perform conversion if all inputs are valid--##
            H2O_conversion = H2O * (P_STP / P) * (T / T_STP)
            H2O_STP.append(H2O_conversion)
    
    ##--Convert H2O ppm to RH wrt Water--##
    
    temperature_c = temperature - 273.15
    
    ##--Lowe and Ficke (1974) 6th deg polynomial approach--##
    ##--Sat vap pressure water -50 to 50 C--##
    wa0 = 6.107799961
    wa1 = 4.436518521E-1
    wa2 = 1.428945805E-2
    wa3 = 2.650648471E-4
    wa4 = 3.031240396E-6
    wa5 = 2.034080948E-8
    wa6 = 6.136820929E-11
    
    ##--Generate empty lists for humididy outputs--##
    saturation_humidity_w = []
    relative_humidity_w = []
    
    ##--Calculate saturation humidity in ppmv and relative humidity--##
    for T, P, H2O in zip(temperature_c, pressure, H2O_data):
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
            RH = (H2O / w_s_ppmv) * 100  # in %
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
    for T, P, H2O in zip(temperature_c, pressure, H2O_data):
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
            RH_i = (H2O / e_si_ppmv) * 100  # in %
            relative_humidity_i.append(RH_i)
        else:
            saturation_humidity_i.append(np.nan)  
            relative_humidity_i.append(np.nan)    
    
    #######################################
    ##--Calculate potential temperature--##
    #######################################
    
    ##--Convert absolute temperature to potential temperature--##
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
    ##--Prepare for Binning--##
    ###########################
    
    ##--Creates separate dfs to preserve data--##
    ##--Including nuc_particles downsizes dataset to instances of N3-10. Comment out if full dataset desired--##
    w_df = pd.DataFrame({'PTemp': potential_temp, 'Latitude': latitude, 'Relative_Humidity_w': relative_humidity_w})#, 'nuc_particles': nuc_particles})
    i_df = pd.DataFrame({'PTemp': potential_temp, 'Latitude': latitude, 'Relative_Humidity_i': relative_humidity_i})#, 'nuc_particles': nuc_particles})
    probe_RH_df = pd.DataFrame({'PTemp': potential_temp, 'Latitude': latitude, 'Probe_RH': RH_probe})
    temp_df = pd.DataFrame({'PTemp': potential_temp, 'Latitude': latitude, 'Temperature': temperature})#, 'nuc_particles': nuc_particles})
    
    ##--Drop NaNs to prevent issues with potential_temp floats--##
    clean_w_df = w_df.dropna()
    clean_i_df = i_df.dropna()
    clean_temp_df = temp_df.dropna()
    
    ##--Compute global min/max values across all data BEFORE dropping NaNs--##
    lat_min, lat_max = np.nanmin(latitude), np.nanmax(latitude)
    ptemp_min, ptemp_max = np.nanmin(potential_temp), np.nanmax(potential_temp)
    
    ##--Generate common bin edges using specified number of bins--##
    common_lat_bin_edges = np.linspace(lat_min, lat_max, num_bins_lat + 1)
    common_ptemp_bin_edges = np.linspace(ptemp_min, ptemp_max, num_bins_ptemp + 1)
    
    ##--Make 2D histogram using common bins--##
    RH_w_bin_medians, _, _, _ = binned_statistic_2d(
        clean_w_df['Latitude'], clean_w_df['PTemp'], clean_w_df['Relative_Humidity_w'], 
        statistic='median', bins=[common_lat_bin_edges, common_ptemp_bin_edges])
    
    RH_i_bin_medians, _, _, _ = binned_statistic_2d(
        clean_i_df['Latitude'], clean_i_df['PTemp'], clean_i_df['Relative_Humidity_i'], 
        statistic='median', bins=[common_lat_bin_edges, common_ptemp_bin_edges])
    
    probe_RH_medians, _, _, _ = binned_statistic_2d(
        probe_RH_df['Latitude'], probe_RH_df['PTemp'], probe_RH_df['Probe_RH'], 
        statistic='median', bins=[common_lat_bin_edges, common_ptemp_bin_edges])
    
    temp_bin_medians, _, _, _ = binned_statistic_2d(
        clean_temp_df['Latitude'], clean_temp_df['PTemp'], clean_temp_df['Temperature'], 
        statistic='median', bins=[common_lat_bin_edges, common_ptemp_bin_edges])
    
    ################
    ##--PLOTTING--##
    ################
     
    def plot_curtain(bin_medians, x_edges, y_edges, vmin, vmax, title, cbar_label): #, output_path):
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
    
        ##--Set axis labels and title--##
        ax.set_xlabel("Latitude (°)", fontsize=16)
        ax.set_ylabel("Potential Temperature \u0398 (K)", fontsize=16)
        ax.tick_params(axis='both', labelsize=16)
        ax.set_title(title, fontsize=18)
        ax.set_ylim(250, 310)
        ax.set_xlim(67, 77)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
     
        ##--Save the plot--##
        #plt.savefig(output_path, dpi=600, bbox_inches="tight")
        plt.tight_layout()
        plt.show()
     
    ##--Plot for RH_w --##
    plot_curtain(RH_w_bin_medians, common_lat_bin_edges, common_ptemp_bin_edges, vmin=1, vmax=105,
        title=f"Relative Humidity w.r.t. Water - {flight.replace('Flight', 'Flight ')} ({flight_date})", cbar_label="% Relative Humidity")
        #output_path=f"{output_path}\\CPC3/PTempLatitude/MultiFlights.png")
    
    ##--Plot for RH_i --##
    plot_curtain(RH_i_bin_medians, common_lat_bin_edges, common_ptemp_bin_edges, vmin=1, vmax=105,
        title=f"Relative Humidity w.r.t. Ice - {flight.replace('Flight', 'Flight ')} ({flight_date})", cbar_label="% Relative Humidity")
        #output_path=f"{output_path}\\CPC10/PTempLatitude/MultiFlights.png")
     
    ##--Plot for probe RH--##
    plot_curtain(probe_RH_medians, common_lat_bin_edges, common_ptemp_bin_edges, vmin=1, vmax=105,
        title=f"Probe Relative Humidity - {flight.replace('Flight', 'Flight ')} ({flight_date})", cbar_label="% Relative Humidity")
        #output_path=f"{output_path}\\Nucleating/PTempLatitude/MultiFlights.png")
    
    ##--Plot for temperature--##
    plot_curtain(temp_bin_medians,  common_lat_bin_edges, common_ptemp_bin_edges, vmin=210, vmax=310,
        title=f"Absolute Temperature - {flight.replace('Flight', 'Flight ')} ({flight_date})", cbar_label='Temperature (K)')
    
    ########################
    ##--Diagnostic Plots--##
    ########################
    
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
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 08:42:10 2025

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

#%%

for flight in flights_to_analyze:
    
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
    
    
    #################
    ##--Pull data--##
    #################
    
    ##--Pull csv file containing all data--##
    files = find_files(directory, flight, "FIREACE")
    
    ##--The 1 hz data is always the first file--##
    if files:
        data = pd.read_csv(files[0])
    
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
    
    ##--Drop rows where Latitude or ptemp are NaN--##
    df = df.dropna(subset=['Latitude', 'ptemp'])
    df_averaged = df_averaged.dropna(subset=['Latitude', 'PTemp'])
    
    ##--Drop rows where Latitude or ptemp are negative--##
    df = df[(df['Latitude'] >= 0) & (df['ptemp'] >= 0)]
    df_averaged = df_averaged[(df_averaged['Latitude'] >= 0) & (df_averaged['PTemp'] >= 0)]
    
    ###########################
    ##--Prepare for Binning--##
    ###########################
    
    ##--Compute global min/max values across all data BEFORE dropping NaNs--##
    lat_min, lat_max = np.nanmin(latitude), np.nanmax(latitude)
    ptemp_min, ptemp_max = np.nanmin(potential_temp), np.nanmax(potential_temp)
    
    ##--Generate common bin edges using specified number of bins--##
    common_lat_bin_edges = np.linspace(lat_min, lat_max, num_bins_lat + 1)
    common_ptemp_bin_edges = np.linspace(ptemp_min, ptemp_max, num_bins_ptemp + 1)
    
    ##--Binning for CPC3 data--##
    all_latitudes_CPC3 = np.concatenate([df["Latitude"].values])
    all_ptemps_CPC3 = np.concatenate([df["ptemp"].values])
    all_CPC3_concs = np.concatenate([df["CPC3_conc"].values])
    
    CPC3_bin_medians, _, _, _ = binned_statistic_2d(all_latitudes_CPC3, all_ptemps_CPC3, 
            all_CPC3_concs, statistic="median", bins=[common_lat_bin_edges, common_ptemp_bin_edges])
     
    ##--Binning for CPC10 data--##
    all_latitudes_CPC10 = np.concatenate([df["Latitude"].values])
    all_ptemps_CPC10 = np.concatenate([df["ptemp"].values])
    all_CPC10_concs = np.concatenate([df["CPC10_conc"].values])
     
     
    CPC10_bin_medians, _, _, _ = binned_statistic_2d(all_latitudes_CPC10, all_ptemps_CPC10, 
            all_CPC10_concs, statistic="median", bins=[common_lat_bin_edges, common_ptemp_bin_edges])
     
    ##--Binning for nucleating particle data--##
    all_latitudes_nuc = np.concatenate([df["Latitude"].values])
    all_ptemps_nuc = np.concatenate([df["ptemp"].values])
    all_nuc_particles = np.concatenate([df["nuc_particles"].values])
    
    
    
    ##--Binning for averaged nucleating particle data--##
    all_latitudes_nuc_averaged = np.concatenate([df_averaged["Latitude"].values])
    all_ptemps_nuc_averaged = np.concatenate([df_averaged["PTemp"].values])
    all_nuc_particles_averaged = np.concatenate([df_averaged["n_3_10_STP"].values])
     
    
    nuc_bin_medians, _, _, _ = binned_statistic_2d(all_latitudes_nuc, all_ptemps_nuc, 
            all_nuc_particles, statistic="median", bins=[common_lat_bin_edges, common_ptemp_bin_edges])
    
    ##--Binning for averaged data: n 10-130--##
    all_latitudes_n_10_130 = np.concatenate([df_averaged["Latitude"].values])
    all_ptemps_n_10_130 = np.concatenate([df_averaged["PTemp"].values])
    all_n_10_130 = np.concatenate([df_averaged["n_10_130"].values])
    
    n_10_130_bin_medians, _, _, _ = binned_statistic_2d(all_latitudes_n_10_130, all_ptemps_n_10_130, 
            all_n_10_130, statistic="median", bins=[common_lat_bin_edges, common_ptemp_bin_edges])
    
    ##--Binning for averaged data: total count--##
    all_latitudes_total = np.concatenate([df_averaged["Latitude"].values])
    all_ptemps_total = np.concatenate([df_averaged["PTemp"].values])
    all_total = np.concatenate([df_averaged['Total_particles_STP'].values])
    
    total_bin_medians, _, _, _ = binned_statistic_2d(all_latitudes_total, all_ptemps_total, 
            all_total, statistic="median", bins=[common_lat_bin_edges, common_ptemp_bin_edges])
    
    
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
     
    ##--Plot for CPC3--##
    plot_curtain(CPC3_bin_medians, common_lat_bin_edges, common_ptemp_bin_edges, vmin=1, vmax=2000,
        title=f"Particles >2.5 nm Abundance - {flight.replace('Flight', 'Flight ')} ({flight_date})", cbar_label="Particles >2.5 nm $(Counts/cm^{3})$")
        #output_path=f"{output_path}\\CPC3/PTempLatitude/MultiFlights.png")
    
    ##--Plot for CPC10--##
    plot_curtain(CPC10_bin_medians, common_lat_bin_edges, common_ptemp_bin_edges, vmin=1, vmax=2000,
        title=f"Particles >10 nm Abundance - {flight.replace('Flight', 'Flight ')} ({flight_date})", cbar_label="Particles >10 nm $(Counts/cm^{3})$")
        #output_path=f"{output_path}\\CPC10/PTempLatitude/MultiFlights.png")
     
    ##--Plot for nucleating particles--##
    plot_curtain(nuc_bin_medians, common_lat_bin_edges, common_ptemp_bin_edges, vmin=1, vmax=2000,
        title=f"2.5-10 nm Particle Abundance - {flight.replace('Flight', 'Flight ')} ({flight_date})", cbar_label="2.5-10 nm Particles $(Counts/cm^{3})$")
        #output_path=f"{output_path}\\Nucleating/PTempLatitude/MultiFlights.png")
    
    ##--Plot for n_10_130--##
    plot_curtain(n_10_130_bin_medians, common_lat_bin_edges, common_ptemp_bin_edges, vmin=1, vmax=2000,
        title=f"10-130 nm Particle Abundance - {flight.replace('Flight', 'Flight ')} ({flight_date})", cbar_label='10-130 nm Particles $(Counts/cm^{3})$')
    
    ##--Plot for total count--##
    plot_curtain(total_bin_medians, common_lat_bin_edges, common_ptemp_bin_edges, vmin=1, vmax=2000,
        title=f"Total Particle Abundance - {flight.replace('Flight', 'Flight ')} ({flight_date})", cbar_label='All Particles $(Counts/cm^{3})$')
    
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
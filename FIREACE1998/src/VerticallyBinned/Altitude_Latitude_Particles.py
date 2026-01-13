# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 10:43:59 2025

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
directory = r"C:\Users\repooley\REP_PhD\Arctic_NPF\FIREACE1998\data"

##--Flights to analyze - flights 1-18--##
flights_to_analyze = ["Flight3",  
                      "Flight7", "Flight8", "Flight9", "Flight10", "Flight11", "Flight12",
                      "Flight13", "Flight14", "Flight15", "Flight16", "Flight17", "Flight18"]

##--Set binning for PTemp and Latitude--##
num_bins_lat = 4
num_bins_alt = 8

##--Base output path in directory--##
output_path = r"C:\Users\repooley\REP_PhD\Arctic_NPF\FIRACE1998\data\processed\AltitudeLatitudeBinned"

PCASP_bins_path = r"C:\Users\repooley\REP_PhD\Arctic_NPF\FIREACE1998\data\raw\FIREACE1998_PCASP_bins.csv"


################################
##--Open Files and pull data--##
################################

##--Define function that creates datasets from filenames--##
def find_files(directory, flight, partial_name):
    ##--flight data are stored in a folder called "raw"--##
    flight_dir = os.path.join(directory, "raw", flight)
    search_pattern = os.path.join(flight_dir, f"*{partial_name}*")
    return sorted(glob.glob(search_pattern))

for flight in flights_to_analyze:
    
    ##--'raw' contains a 1hz and 2min datafile, the 1 hz one is always first--##
    data = pd.read_csv(find_files(directory, flight, "FIREACE")[0])
    
    averaged_data = pd.read_csv(find_files(directory, flight, "FIREACE")[1])
    averaged_data = averaged_data.set_index('Time', drop=False)
    
    PCASP_bins = pd.read_csv(PCASP_bins_path)
    
    ##--Pull data variables from file--##
    time = data['Time'] # HHMMSS UTC time
    pressure = data['Pressure'] * 100 # in Pa
    temperature = data['Temperature'] + 273.15 # in K
    RH_probe = data['RH'] # percent wrt water
    altitude = data['Altitude'] # in m (agl?)
    latitude = data['Latitude'] # degrees
    longitude = data['Longitude'] # degrees
    
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
    
    ##--Based on the supplied data, I strongly believe these two variables were swapped--##
    CO2_data = data['H2O'] # just labeled as 'mv' but there's clear pressure dependence
    H2O_data = data['CO2'] # 'mv'
    
    ##--Particle data, 3 and 10 nm cutoffs, respectively--##
    CPC3_data = data['CN3025_corrected'] # Uncorrected data has a flow issue
    CPC10_data = data['CN7610']
    
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
    
    ##--Creates a Pandas dataframe for CPC data--##
    CPC_df = pd.DataFrame({'CPC3_conc':CPC3_conc_STP, 'CPC10_conc': CPC10_conc_STP})
    
    ##--Calculate N3-10 particles--##
    nuc_particles = (df['CPC3_conc'] - df['CPC10_conc'])
    
    ##--Change calculated particle counts less than zero to NaN--##
    nuc_particles = np.where(nuc_particles >= 0, nuc_particles, np.nan)
    
    ##--Add nucleating particles to df--##
    df['nuc_particles'] = nuc_particles
    
    ##--Put N(2.5-10) bin center in a df--##
    n_3_10_center = pd.DataFrame([6.25]) # Mean of 2.5 and 10
    
    ############################
    ##--Normalize PCASP Data--##
    ############################
    
    ##--Calculate dlogDp for each bin in numpy array--##
    dlogDp = np.log(PCASP_upper_bound.values) - np.log(PCASP_lower_bound.values)
    
    ##--Get only particle count data (excluding 'Time')--##
    PCASP_particle_counts = PCASP_data.loc[:, PCASP_new_col_names]
    
    ##--Normalize counts by dividing by dlogDp across all rows--##
    PCASP_dNdlogDp = PCASP_data.divide(dlogDp, axis=1)
    
    ##--Convert to STP!--##
    P_STP = 101325  # Pa
    T_STP = 273.15  # K
    
    ##--Create empty list for OPC particles--##
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
    
    ######################
    ##--Calc N(10-130)--##
    ######################
    
    ##--Create df with UHSAS total counts--##
    PCASP_total = pd.DataFrame({'Time': averaged_data['Time'], 'Total_count': averaged_data['PCTcon']})
    
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
    
    ###########################
    ##--Wrangle binned data--##
    ###########################
    
    ##--Concatenate bin edges--##
    combined_bin_edges = np.concatenate([
        [2.5],      # start of first bin
        [10],       # upper edge of N(2.5-10), also lower of next
        [130],       # upper edge of N(10-130), also lower of next
        PCASP_upper_bound.values,  # PCASP bins continue from 130
    ])
    
    time_averaged = averaged_data['Time']
    
    ##--Calculate time edges for each bin--##
    time_step = time_averaged.iloc[1] - time_averaged.iloc[0]  
    time_edges = np.append(time_averaged, time_averaged.iloc[-1] + time_step)  # length N + 1
    
    ##--Concatenate bin centers and reindex--##
    bin_centers = pd.concat([n_10_130_center, PCASP_bin_center], axis=0).reset_index(drop=True)
    
    ##--Place all binned data in a single df--##
    all_bins_aligned = pd.concat([n_10_130, PCASP_STP], axis=1)
    
    total_particle_count = all_bins_aligned.sum(axis=1, numeric_only=True) 
    
    ###########################
    ##--Create 2D histogram--##
    ###########################
    
    ##--Float type NaNs in potential_temp cannot convert to int, so must be removed--##
    CPC3_df = pd.DataFrame({'Altitude': altitude, 'Latitude': latitude, 'CPC3':CPC3_conc_STP})
    CPC3_clean_df = CPC3_df.dropna()
    
    ##--Make separate dfs to preserve as much data as possible--##
    CPC10_df = pd.DataFrame({'Altitude': altitude, 'Latitude': latitude, 'CPC10': CPC10_conc_STP})
    CPC10_clean_df = CPC10_df.dropna()
    
    ##--2.5-10 nm, nucleating--##
    nuc_df = pd.DataFrame({'Altitude': altitude, 'Latitude': latitude, 'nuc_particles': nuc_particles})
    nuc_clean_df = nuc_df.dropna()
    
    ##--Compute global min/max values across all data BEFORE dropping NaNs--##
    lat_min, lat_max = np.nanmin(latitude), np.nanmax(latitude)
    alt_min, alt_max = np.nanmin(altitude), np.nanmax(altitude)
    
    ##--Generate common bin edges using specified number of bins--##
    common_lat_bin_edges = np.linspace(lat_min, lat_max, num_bins_lat + 1)
    common_alt_bin_edges = np.linspace(alt_min, alt_max, num_bins_alt + 1)
    
    ##--Make 2D histograms using common bins--##
    ##--CPC3--##
    CPC3_bin_medians, _, _, _ = binned_statistic_2d(CPC3_clean_df['Latitude'], 
        CPC3_clean_df['Altitude'], CPC3_clean_df['CPC3'], statistic='median', bins=[common_lat_bin_edges, common_alt_bin_edges])
    
    ##--CPC10--##
    CPC10_bin_medians, _, _, _ = binned_statistic_2d(CPC10_clean_df['Latitude'], 
        CPC10_clean_df['Altitude'], CPC10_clean_df['CPC10'], statistic='median', bins=[common_lat_bin_edges, common_alt_bin_edges])
    
    ##--N(2.5-10)--##
    nuc_bin_medians, _, _, _ = binned_statistic_2d(nuc_clean_df['Latitude'], 
        nuc_clean_df['Altitude'], nuc_clean_df['nuc_particles'], statistic='median', bins=[common_lat_bin_edges, common_alt_bin_edges])
    
    
    ##--Total count--##
    ##--Float type NaNs in potential_temp cannot convert to int, so must be removed--##
    Count_df = pd.DataFrame({'Altitude': averaged_data['Altitude'], 'Latitude': averaged_data['Latitude'], 
                                   'Count': total_particle_count})
    Count_clean_df = Count_df.dropna()
    
    ##--Compute global min/max values across all data BEFORE dropping NaNs--##
    lat_min, lat_max = np.nanmin(latitude), np.nanmax(latitude)
    alt_min, alt_max = np.nanmin(averaged_data['Altitude']), np.nanmax(averaged_data['Altitude'])
    
    ##--Generate common bin edges using specified number of bins--##
    common_lat_bin_edges = np.linspace(lat_min, lat_max, num_bins_lat + 1)
    common_alt_bin_edges_2 = np.linspace(alt_min, alt_max, num_bins_alt + 1)
    
    ##--Make 2D histograms using common bins--##
    Count_bin_medians, _, _, _ = binned_statistic_2d(Count_clean_df['Latitude'], 
        Count_clean_df['Altitude'], Count_clean_df['Count'], statistic='median', 
        bins=[common_lat_bin_edges, common_alt_bin_edges_2])
    
    ################
    ##--PLOTTING--##
    ################
    
    ##--Particles larger than 3 nm--##
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    
    ##--Make special color map where 0 values are white--##
    new_cmap = plt.get_cmap('viridis')
    ##--Values under specified minimum will be white--##
    new_cmap.set_under('w')
    
    ##--Use pcolormesh for the plot, set minimum value for viridis colors as 1--##
    CPC3_plot = ax1.pcolormesh(common_lat_bin_edges, common_alt_bin_edges, CPC3_bin_medians.T,  # Transpose to align correctly
        shading='auto', cmap=new_cmap, vmin=0, vmax=3500)
    
    ##--Add colorbar--##
    cb = fig1.colorbar(CPC3_plot, ax=ax1)
    cb.minorticks_on()
    cb.set_label('Particles >2.5 nm $(Counts/cm^{3})$', fontsize=12)
    
    # Set axis labels
    ax1.set_xlabel('Latitude (°)', fontsize=12)
    ax1.set_ylabel('Altitude (m)', fontsize=12)
    ax1.set_title(f"Particles >2.5 nm Abundance - {flight.replace('Flight', 'Flight ')} ({flight_date})")
    #ax1.set_ylim(0, 6250)
    #ax1.set_xlim(79.5, 83.7)
    
    ##--Use f-string to save file with flight# appended--##
    CPC3_output_path = f"{output_path}\\/{flight}"
    #plt.savefig(CPC3_output_path, dpi=600, bbox_inches='tight') 
    
    plt.tight_layout()
    plt.show()
    
    ##--Particles larger than 10 nm--##
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    
    ##--Use pcolormesh for the plot, set minimum for viridis colors as 1--##
    CPC10_plot = ax2.pcolormesh(common_lat_bin_edges, common_alt_bin_edges, CPC10_bin_medians.T,  # Transpose to align correctly
        shading='auto', cmap=new_cmap, vmin=0, vmax=3000)
    
    ##--Add colorbar--##
    cb2 = fig2.colorbar(CPC10_plot, ax=ax2)
    cb2.minorticks_on()
    cb2.set_label('Particles >10 nm $(Counts/cm^{3})$', fontsize=12)
    
    ##--Set axis labels--##
    ax2.set_xlabel('Latitude (°)', fontsize=12)
    ax2.set_ylabel('Altitude (m)', fontsize=12)
    ax2.set_title(f"Particles >10 nm Abundance - {flight.replace('Flight', 'Flight ')} ({flight_date})")
    #ax2.set_ylim(0, 6250)
    #ax2.set_xlim(79.5, 83.7)
    
    ##--Use f-string to save file with flight# appended--##
    CPC10_output_path = f"{output_path}\\/{flight}"
    #plt.savefig(CPC10_output_path, dpi=600, bbox_inches='tight') 
    
    plt.tight_layout()
    plt.show()
    
    ##--Nucleating particles--##
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    
    ##--Use pcolormesh for the plot and use viridis for values greater than 1--##
    nuc_plot = ax3.pcolormesh(common_lat_bin_edges, common_alt_bin_edges, nuc_bin_medians.T,  # Transpose to align correctly
        shading='auto', cmap=new_cmap, vmin=0, vmax=1200)
    
    ##--Add colorbar--##
    cb3 = fig3.colorbar(nuc_plot, ax=ax3)
    cb3.minorticks_on()
    cb.ax.tick_params(labelsize=16)
    cb3.set_label('2.5-10 nm Particles $(Counts/cm^{3})$', fontsize=16)
    
    ##--Set axis labels--##
    ax3.set_xlabel('Latitude (°)', fontsize=16)
    ax3.set_ylabel('Altitude (m)', fontsize=16)
    ax3.tick_params(axis='both', labelsize=16)
    ax3.set_title(f"2.5-10 nm Particle Abundance - {flight.replace('Flight', 'Flight ')} ({flight_date})", fontsize=18)
    #ax3.set_ylim(0, 6250)
    #ax3.set_xlim(79.5, 83.7)
    
    ##--Use f-string to save file with flight# appended--##
    nuc_output_path = f"{output_path}\\/{flight}"
    #plt.savefig(nuc_output_path, dpi=600, bbox_inches='tight') 
    
    plt.tight_layout()
    plt.show()
    
    ##--Particles larger than 3 nm--##
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    
    ##--Make special color map where 0 values are white--##
    new_cmap = plt.get_cmap('viridis')
    ##--Values under specified minimum will be white--##
    new_cmap.set_under('w')
    
    ##--Use pcolormesh for the plot, set minimum value for viridis colors as 1--##
    Count_plot = ax1.pcolormesh(common_lat_bin_edges, common_alt_bin_edges, Count_bin_medians.T,  # Transpose to align correctly
        shading='auto', cmap=new_cmap, vmin=0, vmax=5000)
    
    ##--Add colorbar--##
    cb = fig1.colorbar(Count_plot, ax=ax1)
    cb.minorticks_on()
    cb.ax.tick_params(labelsize=16)
    cb.set_label('Total Count $(STP/cm^{3})$', fontsize=16)
    
    ##--Set axis labels--##
    ax1.set_xlabel('Latitude (°)', fontsize=16)
    ax1.set_ylabel('Altitude (m)', fontsize=16)
    ax1.tick_params(axis='both', labelsize=16)
    ax1.set_title(f"Total Particle Count - {flight.replace('Flight', 'Flight ')} ({flight_date})", fontsize=18)
    #ax1.set_ylim(245, 301)
    #ax1.set_xlim(79.5, 83.7)
    
    ##--Use f-string to save file with flight# appended--##
    #Count_output_path = f"{output_path}\\{flight}"
    #plt.savefig(Count_output_path, dpi=600, bbox_inches='tight') 
    
    plt.tight_layout()
    plt.show()
    
    ########################
    ##--Diagnostic Plots--##
    ########################
    
    ##--Remove hashtags below to comment out this section--##
    #'''
    
    ##--Counts per bin for CPC3 data--##
    CPC3_bin_counts, _, _, _ = binned_statistic_2d(CPC3_clean_df['Latitude'], 
        CPC3_clean_df['Altitude'], CPC3_clean_df['CPC3'], statistic='count',
        bins=[common_lat_bin_edges, common_alt_bin_edges])
     
    ##--Counts per bin for CPC10 data--##
    CPC10_bin_counts, _, _, _ = binned_statistic_2d(CPC10_clean_df['Latitude'], 
        CPC10_clean_df['Altitude'], CPC10_clean_df['CPC10'], statistic='count', 
        bins=[common_lat_bin_edges, common_alt_bin_edges])
     
    ##--Counts per bin for N3-10 particles--##
    nuc_bin_counts, _, _, _ = binned_statistic_2d(nuc_clean_df['Latitude'], 
        nuc_clean_df['Altitude'], nuc_clean_df['nuc_particles'], statistic='count', 
        bins=[common_lat_bin_edges, common_alt_bin_edges])
    
    
    ##--Plotting--##
    
    ##--Particles larger than 3 nm--##
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    
    ##--Make special color map where 0 values are white--##
    new_cmap = plt.get_cmap('inferno')
    ##--Values under specified minimum will be white--##
    new_cmap.set_under('w')
    
    ##--Use pcolormesh for the plot, set minimum value for viridis colors as 1--##
    CPC3_diag_plot = ax1.pcolormesh(common_lat_bin_edges, common_alt_bin_edges, CPC3_bin_counts.T,  # Transpose to align correctly
        shading='auto', cmap=new_cmap, vmin=1, vmax=2000)
    
    ##--Add colorbar--##
    cb = fig1.colorbar(CPC3_diag_plot, ax=ax1)
    cb.minorticks_on()
    cb.set_label('Number of Data Points', fontsize=12)
    
    # Set axis labels
    ax1.set_xlabel('Latitude (°)', fontsize=12)
    ax1.set_ylabel('Altitude (m)', fontsize=12)
    ax1.set_title(f"Particles >2.5 nm Counts per Bin - {flight.replace('Flight', 'Flight ')} ({flight_date})")
    #ax1.set_ylim(0, 6250)
    #ax1.set_xlim(79.5, 83.7)
    
    ##--Use f-string to save file with flight# appended--##
    CPC3_diag_output_path = f"{output_path}\\CPC3/AltitudeLatitude/{flight}_diagnostic"
    #plt.savefig(CPC3_diag_output_path, dpi=600, bbox_inches='tight') 
    
    plt.tight_layout()
    plt.show()
    
    ##--Particles larger than 10 nm--##
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    
    ##--Use pcolormesh for the plot, set minimum for viridis colors as 1--##
    CPC10_diag_plot = ax2.pcolormesh(common_lat_bin_edges, common_alt_bin_edges, CPC10_bin_counts.T,  # Transpose to align correctly
        shading='auto', cmap=new_cmap, vmin=1, vmax=2000)
    
    ##--Add colorbar--##
    cb2 = fig2.colorbar(CPC10_diag_plot, ax=ax2)
    cb2.minorticks_on()
    cb2.set_label('Number of Data Points', fontsize=12)
    
    ##--Set axis labels--##
    ax2.set_xlabel('Latitude (°)', fontsize=12)
    ax2.set_ylabel('Altitude (m)', fontsize=12)
    ax2.set_title(f"Particles >10 nm Counts per Bin - {flight.replace('Flight', 'Flight ')} ({flight_date})")
    #ax2.set_ylim(0, 6250)
    #ax2.set_xlim(79.5, 83.7)
    
    ##--Use f-string to save file with flight# appended--##
    CPC10_diag_output_path = f"{output_path}\\CPC10/AltitudeLatitude/{flight}_diagnostic"
    #plt.savefig(CPC10_diag_output_path, dpi=600, bbox_inches='tight') 
    
    plt.tight_layout()
    plt.show()
    
    ##--Nucleating particles--##
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    
    ##--Use pcolormesh for the plot and use viridis for values greater than 1--##
    nuc_diag_plot = ax3.pcolormesh(common_lat_bin_edges, common_alt_bin_edges, nuc_bin_counts.T,  # Transpose to align correctly
        shading='auto', cmap=new_cmap, vmin=1, vmax=2000)
    
    ##--Add colorbar--##
    cb3 = fig3.colorbar(nuc_diag_plot, ax=ax3)
    cb3.minorticks_on()
    cb3.set_label('Number of Data Points', fontsize=12)
    
    ##--Set axis labels--##
    ax3.set_xlabel('Latitude (°)', fontsize=12)
    ax3.set_ylabel('Altitude (m)', fontsize=12)
    ax3.set_title(f"2.5-10 nm Particle Counts per Bin - {flight.replace('Flight', 'Flight ')} ({flight_date})")
    #ax3.set_ylim(0, 6250)
    #ax3.set_xlim(79.5, 83.7)
    
    ##--Use f-string to save file with flight# appended--##
    nuc_diag_output_path = f"{output_path}\\Nucleating/AltitudeLatitude/{flight}_diagnostic"
    #plt.savefig(nuc_diag_output_path, dpi=600, bbox_inches='tight') 
    
    plt.tight_layout()
    plt.show()
    

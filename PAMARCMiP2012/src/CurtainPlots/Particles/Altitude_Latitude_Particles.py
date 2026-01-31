# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 14:05:19 2026

@author: repooley
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from scipy.stats import binned_statistic_2d
from datetime import date

##################
##--Open Files--##
##################

##--Set the base directory to project folder--##
directory = r"C:\Users\repooley\REP_PhD\Arctic_NPF\PAMARCMiP2012\data"

##--Select flight (Flight1 thru Flight9)--##
flights_to_analyze = ["Flight1", "Flight2", "Flight3", "Flight6", 
                      "Flight7", "Flight8", "Flight9"]

##--Set binning for Altitude and Latitude--##
num_bins_lat = 8
num_bins_alt = 8

##--Bin data are in a CSV file--##
##--SAME bins for NETCARE and PAMARCMiP--##
UHSAS_bins = pd.read_csv(r"C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data\raw\NETCARE2015_UHSAS_bins.csv")

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
    
    ###########################
    ##--Create 2D histogram--##
    ###########################
    
    ##--Compute global min/max values across all data BEFORE dropping NaNs--##
    lat_min, lat_max = np.nanmin(latitude), np.nanmax(latitude)
    alt_min, alt_max = np.nanmin(altitude), np.nanmax(altitude)
    
    ##--Generate common bin edges using specified number of bins--##
    common_lat_bin_edges = np.linspace(lat_min, lat_max, num_bins_lat + 1)
    common_alt_bin_edges = np.linspace(alt_min, alt_max, num_bins_alt + 1)
    
    ##--Make 2D histograms using common bins--##
    ##--CPC10--##
    CPC10_bin_medians, _, _, _ = binned_statistic_2d(CPC10_df['Latitude'], 
        CPC10_df['Altitude'], CPC10_df['CPC10_conc'], statistic='median', bins=[common_lat_bin_edges, common_alt_bin_edges])
    
    ##--N(10-60)--##
    grow_bin_medians, _, _, _ = binned_statistic_2d(grow_df['Latitude'],
        grow_df['Altitude'], grow_df['n_10_60'], statistic='median', 
        bins=[common_lat_bin_edges, common_alt_bin_edges])

    ##--Total count--##
    ##--Float type NaNs in potential_temp cannot convert to int, so must be removed--##
    Count_df = pd.DataFrame({'Altitude': altitude, 'Latitude': latitude, 
                                   'Count': total_particle_count})
    Count_clean_df = Count_df.dropna()
    
    ##--Make 2D histograms using common bins--##
    Count_bin_medians, _, _, _ = binned_statistic_2d(Count_clean_df['Latitude'], 
        Count_clean_df['Altitude'], Count_clean_df['Count'], statistic='median', 
        bins=[common_lat_bin_edges, common_alt_bin_edges])
    
    ################
    ##--PLOTTING--##
    ################

    ##--Make special color map where 0 values are white--##
    new_cmap = plt.get_cmap('viridis')
    ##--Values under specified minimum will be white--##
    new_cmap.set_under('w')
    
    ##--Particles larger than 10 nm--##
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    
    ##--Use pcolormesh for the plot, set minimum for viridis colors as 1--##
    CPC10_plot = ax1.pcolormesh(common_lat_bin_edges, common_alt_bin_edges, CPC10_bin_medians.T,  # Transpose to align correctly
        shading='auto', cmap=new_cmap, vmin=0, vmax=3000)
    
    ##--Add colorbar--##
    cb1 = fig1.colorbar(CPC10_plot, ax=ax1)
    cb1.minorticks_on()
    cb1.set_label('Particles >10 nm $(Counts/cm^{3})$', fontsize=12)
    
    ##--Set axis labels--##
    ax1.set_xlabel('Latitude (°)', fontsize=12)
    ax1.set_ylabel('Altitude (m)', fontsize=12)
    ax1.set_title(f"Particles >10 nm Abundance - {flight.replace('Flight', 'Flight ')} ({flight_date})")
    #ax1.set_ylim(0, 6250)
    #ax1.set_xlim(79.5, 83.7)
    
    ##--Use f-string to save file with flight# appended--##
    #CPC10_output_path = f"{output_path}\\/{flight}"
    #plt.savefig(CPC10_output_path, dpi=600, bbox_inches='tight') 
    
    plt.tight_layout()
    plt.show()
    
    
    ##--10-60 nm: Aitken mode--##
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    
    ##--Use pcolormesh for the plot, set minimum for viridis colors as 1--##
    grow_plot = ax2.pcolormesh(common_lat_bin_edges, common_alt_bin_edges, grow_bin_medians.T,  # Transpose to align correctly
        shading='auto', cmap=new_cmap, vmin=0, vmax=3000)
    
    ##--Add colorbar--##
    cb2 = fig2.colorbar(grow_plot, ax=ax2)
    cb2.minorticks_on()
    cb2.set_label('Particles 10-60 nm $(Counts/cm^{3})$', fontsize=12)
    
    ##--Set axis labels--##
    ax2.set_xlabel('Latitude (°)', fontsize=12)
    ax2.set_ylabel('Altitude (m)', fontsize=12)
    ax2.set_title(f"Particles 10-60 nm Abundance - {flight.replace('Flight', 'Flight ')} ({flight_date})")
    #ax2.set_ylim(0, 6250)
    #ax2.set_xlim(79.5, 83.7)
    
    ##--Use f-string to save file with flight# appended--##
    #CPC10_output_path = f"{output_path}\\/{flight}"
    #plt.savefig(CPC10_output_path, dpi=600, bbox_inches='tight') 
    
    plt.tight_layout()
    plt.show()
    
    ##--10-60 nm: Aitken mode--##
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    
    ##--Use pcolormesh for the plot, set minimum for viridis colors as 1--##
    count_plot = ax3.pcolormesh(common_lat_bin_edges, common_alt_bin_edges, Count_bin_medians.T,  # Transpose to align correctly
        shading='auto', cmap=new_cmap, vmin=0, vmax=3000)
    
    ##--Add colorbar--##
    cb3 = fig3.colorbar(count_plot, ax=ax3)
    cb3.minorticks_on()
    cb3.set_label('Total Particle Count $(Counts/cm^{3})$', fontsize=12)
    
    ##--Set axis labels--##
    ax3.set_xlabel('Latitude (°)', fontsize=12)
    ax3.set_ylabel('Altitude (m)', fontsize=12)
    ax3.set_title(f"Total Particle Count - {flight.replace('Flight', 'Flight ')} ({flight_date})")
    #ax3.set_ylim(0, 6250)
    #ax3.set_xlim(79.5, 83.7)
    
    ##--Use f-string to save file with flight# appended--##
    #CPC10_output_path = f"{output_path}\\/{flight}"
    #plt.savefig(CPC10_output_path, dpi=600, bbox_inches='tight') 
    
    plt.tight_layout()
    plt.show()
    

    ########################
    ##--Diagnostic Plots--##
    ########################
    
    ##--Remove hashtags below to comment out this section--##
    '''
    
    ##--Counts per bin for CPC10 data--##
    CPC10_bin_counts, _, _, _ = binned_statistic_2d(CPC10_clean_df['Latitude'], 
        CPC10_clean_df['Altitude'], CPC10_clean_df['CPC10'], statistic='count', 
        bins=[common_lat_bin_edges, common_alt_bin_edges])
     
    ##--Counts per bin for N3-10 particles--##
    nuc_bin_counts, _, _, _ = binned_statistic_2d(nuc_clean_df['Latitude'], 
        nuc_clean_df['Altitude'], nuc_clean_df['nuc_particles'], statistic='count', 
        bins=[common_lat_bin_edges, common_alt_bin_edges])
    
    
    ##--Plotting--##
    
    ##--Make special color map where 0 values are white--##
    new_cmap = plt.get_cmap('inferno')
    ##--Values under specified minimum will be white--##
    new_cmap.set_under('w')
    
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
    #CPC10_diag_output_path = f"{output_path}\\CPC10/AltitudeLatitude/{flight}_diagnostic"
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
    #nuc_diag_output_path = f"{output_path}\\Nucleating/AltitudeLatitude/{flight}_diagnostic"
    #plt.savefig(nuc_diag_output_path, dpi=600, bbox_inches='tight') 
    
    plt.tight_layout()
    plt.show()
    '''
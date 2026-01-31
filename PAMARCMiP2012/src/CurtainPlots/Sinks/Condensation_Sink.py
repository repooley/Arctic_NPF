# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 11:32:11 2026

@author: repooley
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic_2d
import matplotlib.pyplot as plt 
from datetime import date

###################
##--User inputs--##
###################

##--Set the base directory to project folder--##
directory = r"C:\Users\repooley\REP_PhD\Arctic_NPF\PAMARCMiP2012\data\raw"

##--Select flight (Flight1 thru Flight9)--##
flights_to_analyze = ["Flight1", "Flight2", "Flight3", "Flight6", 
                      "Flight7", "Flight8", "Flight9"]

##--Set binning for Altitude and Latitude--##
num_bins_lat = 8
num_bins_ptemp = 8

##--Bin data are in a CSV file--##
##--SAME bins for NETCARE and PAMARCMiP--##
UHSAS_bins = pd.read_csv(r"C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data\raw\NETCARE2015_UHSAS_bins.csv")

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
    ##--Wrangle binned data--##
    ###########################
    
    ##--Concatenate bin edges--##
    combined_bin_edges = np.concatenate([
        [10],       # start of first bin
        [60],       # upper edge of N(10-60), also lower of next
        UHSAS_upper_bound.values,  # UHSAS bins continue from 60
    ])
    
    time_averaged = data['Time']
    
    ##--Calculate time edges for each bin--##
    time_step = time_averaged.iloc[1] - time_averaged.iloc[0]  
    time_edges = np.append(time_averaged, time_averaged.iloc[-1] + time_step)  # length N + 1
    
    ##--Concatenate bin centers and reindex--##
    bin_centers = pd.concat([n_10_60_center, UHSAS_bin_center], axis=0).reset_index(drop=True)
    
    ##--Place all binned data in a single df--##
    all_bins_aligned = UHSAS_bins_filtered
    all_bins_aligned['35'] = n_10_60
    
    time_index = data['Time']  # use the same index as coagulation_sink
    
    ##--Ensure particle bin dataframes are indexed to time_index and properly named--##
    diameter_dfs = {}
    for col in all_bins_aligned.columns:
        vals = all_bins_aligned[col].to_numpy()
        # explicitly name the column as the bin diameter
        diameter_dfs[col] = pd.DataFrame({str(col): vals}, index=time_index[-len(vals):])
        
    ######################################
    ##--Condensation sink calculations--##
    ######################################
    
    ##--Constants--##
    
    R = 8.314 # Ideal gas constant (m^3*Pa*K^-1*mol^-1)
    ##--H2SO4 kinetic diam: lifted from Williamson et al for now (avg of their values)--##
    Ds = 5.49E-10 # in m
    ##--Kinetic diam of air calculated from mixing ratios and dataset on Wikepedia--##
    Dair = 3.61E-10 # in m
    avg_diam = (Ds + Dair)/2
    ##--Mass sulfuric acid--##
    Ms = 98.079 # g/mol
    ##--Mass air--##
    Mair = 28.96 # g/mol
    ##--Reduced mass--##
    Z = Ms/Mair 
    ##--Sticking coefficient - fair to assume unity for H2SO4--##
    alpha = 1
    ##--Boltzmann--##
    k = 1.38E-23 # J/K
    ##--Sutherland's law for dynamic viscosity--##
    C = 1.458E-6 # kg/ms*sqrt(K)
    S = 110.4 # K
    
    ##--Variables--##
    
    ##--Convert temperature and pressure from numpy array to dataframe to subvert errors--##
    ##--Convert temperature and pressure to series with aligned index--##
    temperature_series = pd.Series(temperature.values, index=time_index[-len(temperature):])
    pressure_series = pd.Series(pressure.values, index=time_index[-len(pressure):])
    
    Latitude_series = pd.Series(latitude, index=time_index)
    
    ##--Loop through dfs in diameter_dfs and calculate needed variables for each bin--##
    ##--Store in series initialized at zero--##
    condensation_sink = pd.Series(0, index=time_index)  
    
    for diameter, df in diameter_dfs.items():
        
        ##--Convert column diams from string to float--##
        mean_diameter = (float(diameter)) * 1E-9 # in m
        
        ##--Calculate mean free path of H2SO4 from molecular diameter--##
        df['mean_free_path'] = ((R * temperature_series.loc[df.index]) / ((2 ** (1/2)) 
                                * 3.14159 * (Ds ** 2) * 6.022E23 * pressure_series.loc[df.index])) # in m/molecule
        
        ##--Calculate the Knudsen number--##
        df['Knudsen_num'] = df['mean_free_path'] / (mean_diameter / 2) # unitless ratio
        
        ##--Calculate Fuch's correction--##
        df['Fuchs_correction'] = (1 + df['Knudsen_num']) / (1 + ((4/(3*alpha)) 
                                + 0.337) * df['Knudsen_num'] + (4/(3*alpha) * (df['Knudsen_num']) ** 2)) # unitless
        
        ##--Calculate slip correction for Dynamic Viscosity calculation--##
        df['Slip_correction'] = (1 + df['Knudsen_num'] * (2.514 + 0.800 * (np.exp(-0.550 / df['Knudsen_num'])))) # unitless
        
        ##--Calculate dynamic viscosity using Sutherland's law with constants for air--##
        df['Dynamic_viscosity'] = (C * (temperature_series.loc[df.index]) ** (3/2)) / (temperature_series.loc[df.index] + S)
        
        ##--Calculate the diffusion coefficient--##
        df['Diffusion_coefficient'] = ((k * temperature_series.loc[df.index] * df['Slip_correction']) / 
                                       (3 * 3.14159 * df['Dynamic_viscosity'] * Ds)) # m^2/s
        
        ##--Extract Particle Concentration (first column in diameter_dfs)--##
        df['Particle_concentration'] = df.iloc[:, 0] / 1E-6 # converted to #/m^3
        
        ##--Per-bin contribution to condensation sink (before final multiplication)--##
        df['CS_contribution'] = (df['Fuchs_correction'] * mean_diameter * df['Particle_concentration'])
        
        ##--Multiply each bin’s CS contribution by its diffusion coefficient--##
        ##--Fill NaN values in CS_contribution with zeros to prevent NaN result--##
        condensation_sink += (2 * np.pi * df['Diffusion_coefficient'] * df['CS_contribution']).fillna(0)
     
    ##--Populate series--##
    condensation_sink = pd.DataFrame({'Condensation_Sink': condensation_sink}) 
    
    print((condensation_sink['Condensation_Sink']).max())
    
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
    ##--Create 2D histogram--##
    ###########################
    
    ##--Float type NaNs in potential_temp cannot convert to int, so must be removed--##
    CS10_df = pd.DataFrame({'PTemp': np.asarray(potential_temp), 'Latitude': latitude.to_numpy(), 
                            'CS10': condensation_sink['Condensation_Sink'].to_numpy()}, index=time_index)
    CS10_clean_df = CS10_df.dropna()
    
    ##--Compute global min/max values across all data BEFORE dropping NaNs--##
    lat_min, lat_max = np.nanmin(latitude), np.nanmax(latitude)
    ptemp_min, ptemp_max = np.nanmin(potential_temp), np.nanmax(potential_temp)
    
    ##--Generate common bin edges using specified number of bins--##
    common_lat_bin_edges = np.linspace(lat_min, lat_max, num_bins_lat + 1)
    common_ptemp_bin_edges = np.linspace(ptemp_min, ptemp_max, num_bins_ptemp + 1)
    
    ##--Make 2D histograms using common bins--##
    ##--CPC3--##
    CS10_bin_medians, _, _, _ = binned_statistic_2d(CS10_clean_df['Latitude'], 
        CS10_clean_df['PTemp'], CS10_clean_df['CS10'], statistic='median', bins=[common_lat_bin_edges, common_ptemp_bin_edges])
    
    ################
    ##--PLOTTING--##
    ################
    
    ##--Particles larger than 3 nm--##
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    
    ##--Make special color map where 0 values are white--##
    new_cmap = plt.get_cmap('plasma')
    ##--Values under specified minimum will be white--##
    new_cmap.set_under('w')
    
    ##--Use pcolormesh for the plot, set minimum value for viridis colors as 1--##
    CS10_plot = ax1.pcolormesh(common_lat_bin_edges, common_ptemp_bin_edges, CS10_bin_medians.T,  # Transpose to align correctly
        shading='auto', cmap=new_cmap, vmin=0, vmax=0.0001)
    
    ##--Add colorbar--##
    cb = fig1.colorbar(CS10_plot, ax=ax1)
    cb.minorticks_on()
    cb.ax.tick_params(labelsize=16)
    cb.set_label('CS10 (s-1)', fontsize=16)
    
    ##--Set axis labels--##
    ax1.set_xlabel('Latitude (°)', fontsize=16)
    ax1.set_ylabel('Potential Temperature \u0398 (K)', fontsize=16)
    ax1.tick_params(axis='both', labelsize=16)
    ax1.set_title(f"Condensation Sink - {flight.replace('Flight', 'Flight ')} ({flight_date})", fontsize=18)
    #ax1.set_ylim(250, 310)
    #ax1.set_xlim(67, 77)
    
    ##--Use f-string to save file with flight# appended--##
    #CS10_output_path = f"{output_path}\\{flight}"
    #plt.savefig(CS10_output_path, dpi=600, bbox_inches='tight') 
    
    plt.tight_layout()
    plt.show()
    
    ########################
    ##--Diagnostic Plots--##
    ########################
    
    ##--Remove hashtags below to comment out this section--##
    #'''
    
    ##--Counts per bin for CPC3 data--##
    CS10_bin_counts, _, _, _ = binned_statistic_2d(CS10_clean_df['Latitude'], 
        CS10_clean_df['PTemp'], CS10_clean_df['CS10'], statistic='count', bins=[common_lat_bin_edges, common_ptemp_bin_edges])
    
    ##--Particles larger than 3 nm--##
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    
    ##--Make special color map where 0 values are white--##
    new_cmap = plt.get_cmap('inferno')
    ##--Values under specified minimum will be white--##
    new_cmap.set_under('w')
    
    ##--Use pcolormesh for the plot, set minimum value for viridis colors as 1--##
    CS10_plot = ax1.pcolormesh(common_lat_bin_edges, common_ptemp_bin_edges, CS10_bin_counts.T,  # Transpose to align correctly
        shading='auto', cmap=new_cmap, vmin=1, vmax=50)
    
    
    ##--Add colorbar--##
    cb = fig1.colorbar(CS10_plot, ax=ax1)
    cb.minorticks_on()
    cb.ax.tick_params(labelsize=16)
    cb.set_label('Number of Data Points', fontsize=16)
    
    # Set axis labels
    ax1.set_xlabel('Latitude (°)', fontsize=16)
    ax1.set_ylabel('Potential Temperature \u0398 (K)', fontsize=16)
    ax1.tick_params(axis='both', labelsize=16)
    ax1.set_title(f"Condensation Sink Counts per Bin - {flight.replace('Flight', 'Flight ')} ({flight_date})", fontsize=18)
    #ax1.set_ylim(238, 301)
    #ax1.set_xlim(79.5, 83.7)
    
    ##--Use f-string to save file with flight# appended--##
    #CS10_diag_output_path = f"{output_path}\\{flight}_diagnostic"
    #plt.savefig(CS10_diag_output_path, dpi=600, bbox_inches='tight') 
    
    plt.tight_layout()
    plt.show()
    #'''
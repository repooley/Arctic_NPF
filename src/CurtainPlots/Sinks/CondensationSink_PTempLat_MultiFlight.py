# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 16:38:41 2025

@author: repooley
"""

import icartt
import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic_2d
import matplotlib.pyplot as plt 

###################
##--User inputs--##
###################

##--Set the base directory to project folder--##
directory = r"C:\Users\repooley\REP_PhD\NETCARE2015\data\raw"

##--Choose which flights to analyze here!--##
##--FLIGHT1 HAS NO USHAS FILE--##
flights_to_analyze = ["Flight2", "Flight3", "Flight4", "Flight5", "Flight6", 'Flight7', 'Flight8', 'Flight9', 'Flight10']

##--Set number of bins for latitude and potential temperature--##
num_bins_lat = 10
num_bins_ptemp = 10

##--Base output path for figures in directory--##
output_path = r"C:\Users\repooley\REP_PhD\NETCARE2015\data\processed\CurtainPlots\CondensationSink"

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
##--PTemp is populated as a column in condensation_sinks--##
condensation_sinks = []
 
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
        print(f"No UHSAS_POLAR6 file found for {flight}. Skipping...")
        continue  
    
    ##--Pull OPC files--##
    OPC_files = find_files(flight_dir, 'OPC')
 
    if OPC_files: 
        OPC = icartt.Dataset(OPC_files[0])
    else: 
        print(f'Missing OPC data for {flight}. Skipping...')
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
    
    ##--USHAS Data--##
    UHSAS_time = UHSAS.data['time'] # seconds since midnight

    ##--Bin data are in a CSV file--##
    UHSAS_bins = pd.read_csv(r"C:\Users\repooley\REP_PhD\NETCARE2015\data\raw\NETCARE2015_UHSAS_bins.csv")

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
    
    ##--Tabulate total count across all bins--##
    UHSAS_total_num = UHSAS_bins_aligned.sum(axis=1, numeric_only=True) # particles/cm^3
    
    ##--OPC Data--##
    OPC_time = OPC.data['Time_UTC'] # seconds since midnight

    ##--Bin data are in a CSV file--##
    OPC_bin_info = pd.read_csv(r"C:\Users\repooley\REP_PhD\NETCARE2015\data\raw\NETCARE2015_OPC_bins.csv")
    
    ##--Select bins greater than 500 nm (Channel 7 and greater)--##
    OPC_bin_center = OPC_bin_info['bin_avg'].iloc[6:31]
    OPC_lower_bound = OPC_bin_info['lower_bound'].iloc[6:31]
    OPC_upper_bound = OPC_bin_info['upper_bound'].iloc[6:31]

    ##--Make list of columns to pull, each named Channel_x--##
    OPC_bin_num = [f'Channel_{i}' for i in range(7, 32)]

    ##--Put column names and content in a dictionary and then convert to a Pandas df--##
    OPC_bins = pd.DataFrame({col: OPC.data[col] for col in OPC_bin_num})

    ##--Create new column names by rounding the bin center values to the nearest integer--##
    OPC_new_col_names = OPC_bin_center.round().astype(int).tolist()

    ##--Rename the OPC_bins df columns to bin average values--##
    OPC_bins.columns = OPC_new_col_names

    ##--Add time, total_num to OPC_bins df--##
    OPC_bins.insert(0, 'Time', OPC_time)

    ##--Align OPC_bins time to AIMMS time--##
    OPC_bins_aligned = OPC_bins.set_index('Time').reindex(aimms_time)

    ##--OPC samples every six seconds. Most rows are NaN--##
    ##--Forward-fill NaN values to propagate last valid reading--##
    ##--Limit forward filling to 5 NaN rows--##
    OPC_bins_filled = OPC_bins_aligned.ffill(limit=5)

    ###############################
    ##--De-Normalize UHSAS Data--##
    ###############################

    ##--Calculate dlogDp for UHSAS bins--##
    UHSAS_dlogDp = np.log(UHSAS_upper_bound.values) - np.log(UHSAS_lower_bound.values)

    ##--Get only particle count data (excluding 'Time')--##
    UHSAS_particle_counts = UHSAS_bins_aligned.loc[:, UHSAS_new_col_names]  # Adjust column names as needed

    ##--De-Normalize counts by multiplying by dlogDp across all rows--##
    UHSAS_denorm_counts = UHSAS_particle_counts.multiply(UHSAS_dlogDp, axis=1)

    ##--Take out of STP--##
    P_STP = 101325  # Pa
    T_STP = 273.15  # K

    ##--Create empty list for OPC particles--##
    UHSAS_abs_counts = []

    for UHSAS, T, P in zip(UHSAS_denorm_counts.values, temperature, pressure):
        if np.isnan(T) or np.isnan(P):
            ##--Append with NaN if any input is NaN--##
            UHSAS_abs_counts.append([np.nan]*len(UHSAS))
        else:
            ##--Perform conversion if all inputs are valid--##
            corrected_UHSAS = UHSAS / (P_STP / P) / (T / T_STP)
            UHSAS_abs_counts.append(corrected_UHSAS)

    ##--Convert back to DataFrame with same columns and index--##
    UHSAS_abs_counts = pd.DataFrame(UHSAS_abs_counts, columns=UHSAS_denorm_counts.columns, index=UHSAS_denorm_counts.index)

    ##--Reset index of UHSAS_abs_counts to align with time--##
    min_length = min(len(UHSAS_time), len(UHSAS_abs_counts))
    UHSAS_time = UHSAS_time[:min_length]
    UHSAS_abs_counts = UHSAS_abs_counts.iloc[:min_length]

    #####################
    ##--Calc N(10-89)--##
    #####################

    ##--Create df with UHSAS total counts--##
    UHSAS_total = pd.DataFrame({'Time': UHSAS_time, 'Total_count': UHSAS_abs_counts.sum(axis=1)})

    ##--Reindex UHSAS_total df to AIMMS time--##
    UHSAS_total_aligned = UHSAS_total.set_index('Time').reindex(aimms_time)

    ##--Create df with CPC10 counts and set index to time--##
    CPC10_counts = pd.DataFrame({'Time':aimms_time, 'Counts':CPC10_conc_aligned}).set_index('Time')

    ##--Calculate particles below UHSAS lower cutoff--##
    n_10_89 = (CPC10_counts['Counts'] - UHSAS_total_aligned['Total_count'])

    ##--Change calculated particle counts less than zero to NaN--##
    n_10_89 = np.where(n_10_89 >= 0, n_10_89, np.nan)

    ##--Put N(10-89) bin center in a df--##
    n_10_89_center = pd.DataFrame([49.5])

    ##--Convert n_10_89 to a df--##
    n_10_89 = pd.DataFrame({'49.5': n_10_89, 'time':aimms_time}).set_index('time')

    ##--Change first column name from string to integer--##
    n_10_89.columns = [49.5]

    ###########################
    ##--Wrangle binned data--##
    ###########################
    
    ##--Concatenate bin edges--##
    combined_bin_edges = np.concatenate([
        [2.5],      # start of first bin
        [10],       # upper edge of N(2.5-10), also lower of next
        [89.32],       # upper edge of N(10-89), also lower of next
        UHSAS_upper_bound.values,  # UHSAS bins continue from 85
        OPC_upper_bound.values     # OPC bins continue from last UHSAS
    ])
    
    ##--Calculate time edges for each bin--##
    time_step = aimms_time[1] - aimms_time[0]  
    time_edges = np.append(aimms_time, aimms_time[-1] + time_step)  # length N + 1
    
    ##--Concatenate bin centers and reindex--##
    bin_centers = pd.concat([n_10_89_center, UHSAS_bin_center, OPC_bin_center], axis=0).reset_index(drop=True)
    
    ##--Place all binned data in a single df--##
    all_bins_aligned = pd.concat([n_10_89, UHSAS_bins_aligned, OPC_bins_filled], axis=1)
    total_particle_count = all_bins_aligned.sum(axis=1, numeric_only=True) 
    
    ##--Create a dictionary to store each column as a separate dataframe, col names are keys--##
    diameter_dfs = {col: pd.DataFrame({col: all_bins_aligned[col]}) for col in all_bins_aligned.columns}
    
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
    temperature_series = pd.Series(temperature, index=aimms_time)
    pressure_series = pd.Series(pressure, index=aimms_time)
    
    ##--Loop through dfs in diameter_dfs and calculate needed variables for each bin--##
    ##--Store in series initialized at zero--##
    condensation_sink = pd.Series(0, index=aimms_time)
    
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
    
    ##--Append latitude data--##
    condensation_sink['Latitude'] = latitude
    
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
        
    condensation_sink['PTemp'] = potential_temp
    
    ##--Append condensation_sinks with the columns--##
    condensation_sinks.append(condensation_sink.dropna(subset=['Condensation_Sink', 'Latitude', 'PTemp']))


###########################
##--Create 2D histogram--##
###########################

##--Float type NaNs in potential_temp cannot convert to int, so must be removed--##
all_latitudes_CS10 = np.concatenate([df['Latitude'].values for df in condensation_sinks])
all_ptemps_CS10 = np.concatenate([df['PTemp'].values for df in condensation_sinks])
all_CS10 = np.concatenate([df['Condensation_Sink'].values for df in condensation_sinks])
 
lat_bin_edges_CS10 = np.linspace(all_latitudes_CS10.min(), all_latitudes_CS10.max(), num_bins_lat + 1)
ptemp_bin_edges_CS10 = np.linspace(all_ptemps_CS10.min(), all_ptemps_CS10.max(), num_bins_ptemp + 1)
 
CS10_bin_medians, _, _, _ = binned_statistic_2d(all_latitudes_CS10, all_ptemps_CS10, 
        all_CS10, statistic="mean", bins=[lat_bin_edges_CS10, ptemp_bin_edges_CS10])

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
CS10_plot = ax1.pcolormesh(lat_bin_edges_CS10, ptemp_bin_edges_CS10, CS10_bin_medians.T,  # Transpose to align correctly
    shading='auto', cmap=new_cmap, vmin=0, vmax=0.006)

##--Add dashed horizontal lines for the polar dome boundaries--##
ax1.axhline(y=275, color='k', linestyle='--', linewidth=1)
ax1.axhline(y=299, color='k', linestyle='--', linewidth=1)

##--Add colorbar--##
cb = fig1.colorbar(CS10_plot, ax=ax1)
cb.minorticks_on()
cb.ax.tick_params(labelsize=16)
cb.set_label('CS10 (s-1)', fontsize=16)

##--Set axis labels--##
ax1.set_xlabel('Latitude (°)', fontsize=16)
ax1.set_ylabel('Potential Temperature \u0398 (K)', fontsize=16)
ax1.tick_params(axis='both', labelsize=16)
ax1.set_title("Condensation Sink", fontsize=18)
#ax1.set_ylim(238, 301)
#ax1.set_xlim(79.5, 83.7)

##--Use f-string to save file with flight# appended--##
CS10_output_path = f"{output_path}\\{flight}_MultiFlights"
plt.savefig(CS10_output_path, dpi=600, bbox_inches='tight') 

plt.tight_layout()
plt.show()

########################
##--Diagnostic Plots--##
########################

##--Remove hashtags below to comment out this section--##
#'''

##--Counts per bin for CPC3 data--##
CS10_bin_counts, _, _, _ = binned_statistic_2d(all_latitudes_CS10, 
    all_ptemps_CS10, all_CS10, statistic='count', bins=[lat_bin_edges_CS10, ptemp_bin_edges_CS10])

##--Particles larger than 3 nm--##
fig1, ax1 = plt.subplots(figsize=(8, 6))

##--Make special color map where 0 values are white--##
new_cmap = plt.get_cmap('inferno')
##--Values under specified minimum will be white--##
new_cmap.set_under('w')

##--Use pcolormesh for the plot, set minimum value for viridis colors as 1--##
CS10_plot = ax1.pcolormesh(lat_bin_edges_CS10, ptemp_bin_edges_CS10, CS10_bin_counts.T,  # Transpose to align correctly
    shading='auto', cmap=new_cmap, vmin=1, vmax=10000)

##--Add dashed horizontal lines for the polar dome boundaries--##
ax1.axhline(y=275, color='k', linestyle='--', linewidth=1)
ax1.axhline(y=299, color='k', linestyle='--', linewidth=1)

##--Add colorbar--##
cb = fig1.colorbar(CS10_plot, ax=ax1)
cb.minorticks_on()
cb.ax.tick_params(labelsize=16)
cb.set_label('Number of Data Points', fontsize=16)

# Set axis labels
ax1.set_xlabel('Latitude (°)', fontsize=16)
ax1.set_ylabel('Potential Temperature \u0398 (K)', fontsize=16)
ax1.tick_params(axis='both', labelsize=16)
ax1.set_title("Condensation Sink Counts per Bin", fontsize=18)
#ax1.set_ylim(238, 301)
#ax1.set_xlim(79.5, 83.7)

##--Use f-string to save file with flight# appended--##
CS10_diag_output_path = f"{output_path}\\MultiFlights_diagnostic"
plt.savefig(CS10_diag_output_path, dpi=600, bbox_inches='tight') 

plt.tight_layout()
plt.show()
#'''
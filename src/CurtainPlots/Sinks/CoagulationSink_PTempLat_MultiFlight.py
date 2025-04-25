# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 09:59:06 2025

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
##--NO UHSAS FLIGHT1--##
flights_to_analyze = ["Flight2", "Flight3", "Flight4", "Flight5", "Flight6", 'Flight7', 'Flight8', 'Flight9', 'Flight10']

##--Set number of bins for latitude and potential temperature--##
num_bins_lat = 10
num_bins_ptemp = 10

##--Base output path for figures in directory--##
output_path = r"C:\Users\repooley\REP_PhD\NETCARE2015\data\processed\CurtainPlots\CoagulationSink"

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
coagulation_sinks = []
 
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
    UHSAS_total_num = UHSAS_bins_aligned.sum(axis=1, numeric_only=True)

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
    
    #####################################
    ##--Coagulation sink calculations--##
    #####################################

    ##--Constants--##
    
    ##--Ideal gas--##
    R = 8.314 # m^3*Pa*K^-1*mol^-1
    ##--Boltzmann--##
    k = 1.38E-23 # m^2*kg*s^-2*K^-1
    ##--Sutherland's law for dynamic viscosity--##
    C = 1.458E-6 # kg/ms*sqrt(K)
    S = 110.4 # K
    ##--Molar mass air molecules--##
    MMair = 28.96 # g/mol
    ##--Mass single air molecule in kg--##
    Mair = MMair/(6.022E23 * 1000) # kg
    ##--Diameter average air molecule--##
    Dair = 3.61E-10 # in m
    
    ##--For N(2.5-10)--##
    
    ##--Median particle diameter--##
    nuc_diam = 6.25E-9 # m
    
    ##--Vol spherical particle--##
    nuc_vol = (4/3) * np.pi * (nuc_diam / 2) ** 3 #m^3
    
    ##--Mass particle assuming density of 1--##
    nuc_mass = nuc_vol #kg
    
    ##--Reduced mass ratio--##
    z_nuc = nuc_mass/Mair
    
    ##--Collision cross section nuc particle + air--##
    sigma_nuc = (Dair + nuc_diam) / 2
    
    ##--Variables--##
    
    ##--Convert temperature and pressure from numpy array to dataframe to subvert errors--##
    temperature_series = pd.Series(temperature, index=aimms_time)
    pressure_series = pd.Series(pressure, index=aimms_time)
    ##--Concentration air molecules--##
    Nair = (6.022E23 * pressure) / (R * temperature) # num/m^3
    
    ##--For N(2.5-10)--##
    
    ##--Mean particle speed--##
    nuc_speed = ((8 * k * temperature_series) / (np.pi * nuc_mass))**(1/2)
    
    ##--Collision cross section, assumes collision with equal size particle--##
    nuc_collision_cross = np.pi * (nuc_diam )**2 # m^2
    
    ##--Estimate of mean free path against air for use in slip correction--##
    nuc_mfp_estimate = 1/(np.pi * (1 + z_nuc)**(1/2) * Nair * sigma_nuc**2) # m
    
    ##--Knudsen number--##
    nuc_knudsen = nuc_mfp_estimate / (nuc_diam/2) # unitless 
    
    ##--Cunningham slip correction--##
    nuc_slip = 1 + 2 * nuc_knudsen * (2.514 + 0.800 * np.exp(-0.550 / nuc_knudsen))
    
    ##--Dynamic viscosity--##
    dynam_viscosity = (C * temperature_series ** (3/2)) / (temperature_series + S)
    
    ##--Diffusivity--##
    nuc_diffusivity = (k * temperature_series * nuc_slip / (3 * np.pi * dynam_viscosity * nuc_diam)) # m^2/s
    
    ##--Mean free path--##
    nuc_mfp = ((8 * nuc_diffusivity) / (np.pi * nuc_speed)) # m 
    
    ##--Calculate g coefficient--##
    nuc_g = (2**(1/2) / (3 * nuc_diam * nuc_mfp)) * ((nuc_diam + nuc_mfp)**3 - (nuc_diam**2 + nuc_mfp**2)**(3/2)) - nuc_diam
    
    
    ##--Loop through dfs in diameter_dfs and calculate needed variables for each bin--##
    ##--Store in series starting at zero--##
    coagulation_sink = pd.Series(0, index=aimms_time) 
    
    for diameter, df in diameter_dfs.items():
        
        ##--Convert column diams from string to float--##
        mean_diameter = (float(diameter)) * 1E-9 # in m
        
        ##--Particle volume per diameter bin--##
        volume = (4/3) * np.pi * (nuc_diam / 2) ** 3 #m^3
        
        ##--Particle mass, assuming a density of 1--##
        mass = volume # kg
        
        ##--Particle speed--##
        speed = ((8 * k * temperature_series) / (np.pi * mass))**(1/2)
        
        ##--Reduced mass ratio--##
        z = mass/Mair
        
        ##--Collision cross section air + particle--##
        sigma = (mean_diameter + Dair) / 2
        
        ##--Extract Particle Concentration (first column in diameter_dfs)--##
        df['Particle_concentration'] = df.iloc[:, 0] / 1E-6 # converted to #/m^3
        
        ##--Estimate of mean free path (against collision with air) for use in slip correction--##
        mfp_estimate = 1/(np.pi * (1 + z)**(1/2) * Nair * sigma**2) # m
        
        ##--Knudsen number--##
        df['Knudsen_number'] = mfp_estimate / (mean_diameter / 2) # unitless
        
        ##--Cunningham slip correction in air--##
        slip = 1 + 2 * df['Knudsen_number'] * (2.514 + 0.800 * np.exp(-0.550 / df['Knudsen_number'])) # unitless
        
        ##--Particle diffusivity--##
        diffusivity = (k * temperature_series * nuc_slip / (3 * np.pi * dynam_viscosity * mean_diameter)) # m^2/s
        
        ##--Calculate mean free path of H2SO4 from molecular diameter--##
        df['mean_free_path'] = ((8 * diffusivity) / (np.pi * speed)) # m
        
        ##--Calculate g coefficient--##
        g = ((2**(1/2) / (3 * mean_diameter * df['mean_free_path'])) * ((mean_diameter + df['mean_free_path'])**3 
                                          - (mean_diameter**2 + df['mean_free_path']**2)**(3/2)) - mean_diameter)
        
        ##--Compute the coagulation kernel per bin--##
        df['Coagulation_kernel'] = (2 * np.pi * (nuc_diffusivity + diffusivity) * (nuc_diam + mean_diameter) * 
                              ((nuc_diam + mean_diameter) / (nuc_diam + mean_diameter + 2*(nuc_g**2 + g**2)**(1/2))
                               + (8 * (nuc_diffusivity + diffusivity)) / ((nuc_speed**2 + speed**2)**(1/2) * 
                                                                          (nuc_diam + mean_diameter)))**-1)
        
        ##--Calculate coagulation by multiplying kernel by particle concentration per bin--##
        df['Coagulation'] = df['Coagulation_kernel'] * df['Particle_concentration'] # s^-1
        
        ##--Sum the coagulation kernels across bins--##
        ##--Replace NaN values with zero (no contribution)--##
        coagulation_sink += (df['Coagulation']).fillna(0)  

    ##--Populate series--##
    coagulation_sink = pd.DataFrame({'Coagulation': coagulation_sink}) 
    
    ##--Append latitude data--##
    coagulation_sink['Latitude'] = latitude

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

    coagulation_sink['PTemp'] = potential_temp

    ##--Append condensation_sinks with the columns--##
    coagulation_sinks.append(coagulation_sink.dropna(subset=['Coagulation', 'Latitude', 'PTemp']))

###########################
##--Create 2D histogram--##
###########################

##--Float type NaNs in potential_temp cannot convert to int, so must be removed--##
all_latitudes_coagulation = np.concatenate([df['Latitude'].values for df in coagulation_sinks])
all_ptemps_coagulation = np.concatenate([df['PTemp'].values for df in coagulation_sinks])
all_coagulation = np.concatenate([df['Coagulation'].values for df in coagulation_sinks])
 
lat_bin_edges_coagulation = np.linspace(all_latitudes_coagulation.min(), all_latitudes_coagulation.max(), num_bins_lat + 1)
ptemp_bin_edges_coagulation = np.linspace(all_ptemps_coagulation.min(), all_ptemps_coagulation.max(), num_bins_ptemp + 1)
 
coagulation_bin_medians, _, _, _ = binned_statistic_2d(all_latitudes_coagulation, all_ptemps_coagulation, 
        all_coagulation, statistic="mean", bins=[lat_bin_edges_coagulation, ptemp_bin_edges_coagulation])

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
Coagulation_plot = ax1.pcolormesh(lat_bin_edges_coagulation, ptemp_bin_edges_coagulation, coagulation_bin_medians.T,  # Transpose to align correctly
    shading='auto', cmap=new_cmap, vmin=0, vmax=0.0014)

##--Add dashed horizontal lines for the polar dome boundaries--##
ax1.axhline(y=285, color='k', linestyle='--', linewidth=1)
ax1.axhline(y=299, color='k', linestyle='--', linewidth=1)

##--Add colorbar--##
cb = fig1.colorbar(Coagulation_plot, ax=ax1)
cb.minorticks_on()
cb.ax.tick_params(labelsize=16)
cb.set_label('Coagulation N(2.5-10) (s-1)', fontsize=16)

##--Set axis labels--##
ax1.set_xlabel('Latitude (°)', fontsize=16)
ax1.set_ylabel('Potential Temperature \u0398 (K)', fontsize=16)
ax1.tick_params(axis='both', labelsize=16)
ax1.set_title("Coagulation Sink N(2.5-10)", fontsize=18)
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
Coagulation_bin_counts, _, _, _ = binned_statistic_2d(all_latitudes_coagulation, 
    all_ptemps_coagulation, all_coagulation, statistic='count', bins=[lat_bin_edges_coagulation, ptemp_bin_edges_coagulation])

##--Particles larger than 3 nm--##
fig1, ax1 = plt.subplots(figsize=(8, 6))

##--Make special color map where 0 values are white--##
new_cmap = plt.get_cmap('inferno')
##--Values under specified minimum will be white--##
new_cmap.set_under('w')

##--Use pcolormesh for the plot, set minimum value for viridis colors as 1--##
Coagulation_plot = ax1.pcolormesh(lat_bin_edges_coagulation, ptemp_bin_edges_coagulation, Coagulation_bin_counts.T,  # Transpose to align correctly
    shading='auto', cmap=new_cmap, vmin=1, vmax=10000)

##--Add dashed horizontal lines for the polar dome boundaries--##
ax1.axhline(y=285, color='k', linestyle='--', linewidth=1)
ax1.axhline(y=299, color='k', linestyle='--', linewidth=1)

##--Add colorbar--##
cb = fig1.colorbar(Coagulation_plot, ax=ax1)
cb.minorticks_on()
cb.ax.tick_params(labelsize=16)
cb.set_label('Number of Data Points', fontsize=16)

# Set axis labels
ax1.set_xlabel('Latitude (°)', fontsize=16)
ax1.set_ylabel('Potential Temperature \u0398 (K)', fontsize=16)
ax1.tick_params(axis='both', labelsize=16)
ax1.set_title("Coagulation Sink Counts per Bin", fontsize=18)
#ax1.set_ylim(238, 301)
#ax1.set_xlim(79.5, 83.7)

##--Use f-string to save file with flight# appended--##
CS10_diag_output_path = f"{output_path}\\MultiFlights_diagnostic"
plt.savefig(CS10_diag_output_path, dpi=600, bbox_inches='tight') 

plt.tight_layout()
plt.show()
#'''
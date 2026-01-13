# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 08:23:11 2025

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
directory = r"C:\Users\repooley\REP_PhD\Arctic_NPF\FIREACE1998\data\raw"

##--Flights to analyze - flights 1-18--##
flights_to_analyze = ["Flight3",  
                      "Flight7", "Flight8", "Flight9", "Flight10", "Flight11", "Flight12",
                      "Flight13", "Flight14", "Flight15", "Flight16", "Flight17", "Flight18"]

##--Set binning for PTemp and Latitude--##
##--Define number of bins here--##
num_bins_lat = 6
num_bins_ptemp = 12

PCASP_bins_path = r"C:\Users\repooley\REP_PhD\Arctic_NPF\FIREACE1998\data\raw\FIREACE1998_PCASP_bins.csv"

##--Base output path in directory--##
output_path = r"C:\Users\repooley\REP_PhD\Arctic_NPF\FIREACE1998\data\processed"

##################
##--Open Files--##
##################

##--Define function that creates datasets from filenames--##
def find_files(directory, flight, partial_name):
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
    
    ##--The averaged data is always the second file--##
    if files:
        data = pd.read_csv(files[1])
    
    ##--Pull data variables from file--##
    time = data['Time'] # HHMMSS UTC time
    pressure = data['Pressure'] * 100 # in Pa
    temperature = data['Temperature'] + 273.15 # in K
    RH = data['RH'] # percent wrt water
    altitude = data['Altitude'] # in m (agl?)
    latitude = data['Latitude'] # degrees
    #longitude = data['Longitude'] # degrees
    
    ##--Particle data, 3 and 10 nm cutoffs, respectively--##
    CPC3_data = data['CN3025'] # Uncorrected data has a flow issue - but corrected not populated for many flights
    CPC10_data = data['CN7610']
    
    PCASP_bins = pd.read_csv(PCASP_bins_path)
    
    PCASP_data = data.iloc[:, 14:29] # select PCASP data
    
    ##--Add time, total_num to UHSAS_bins df--##
    PCASP_data.insert(0, 'Time', data['Time'])
    
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
    particle_df = pd.DataFrame({'Altitude': altitude, 'Latitude': latitude,
                       'CPC3_conc':CPC3_conc_STP, 'CPC10_conc': CPC10_conc_STP})
    
    ##--Calculate N3-10 particles--##
    nuc_particles = (particle_df['CPC3_conc'] - particle_df['CPC10_conc'])
    
    ##--Change calculated particle counts less than zero to NaN--##
    nuc_particles = np.where(nuc_particles >= 0, nuc_particles, np.nan)
    
    ##--Add nucleating particles to df--##
    particle_df['n_3_10'] = nuc_particles
    
    
    ############################
    ##--Normalize PCASP data--##
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
    
    ##--Create empty list for PCASP particles--##
    PCASP_STP = []
    
    for PCASP, T, P in zip(PCASP_dNdlogDp.values, data['Temperature']+273.15, data['Pressure']*100):
        if np.isnan(T) or np.isnan(P):
            ##--Append with NaN if any input is NaN--##
            PCASP_STP.append([np.nan]*len(PCASP))
        else:
            ##--Perform conversion if all inputs are valid--##
            corrected_PCASP = PCASP * (P_STP / P) * (T / T_STP)
            PCASP_STP.append(corrected_PCASP)
    
    ##--Convert back to DataFrame with same columns and index--##
    PCASP_STP = pd.DataFrame(PCASP_STP, columns=PCASP_dNdlogDp.columns, index=particle_df.index)
    
    ##--Add PCASP data to the dataframe--##
    particle_df = pd.concat([particle_df, PCASP_STP], axis=1)
    
    ##--Add PCASP total counts to the dataframe--##
    particle_df['PCTcon'] = data['PCTcon']
    
    ######################
    ##--Calc N(10-130)--##
    ######################
    
    ##--Calculate particles below UHSAS lower cutoff--##
    n_10_130 = (particle_df['CPC10_conc'] - particle_df['PCTcon'])
    
    ##--Change calculated particle counts less than zero to NaN--##
    n_10_130 = np.where(n_10_130 >= 0, n_10_130, np.nan)
    
    ##--Put N(10-130) bin center in a df--##
    n_10_130_center = pd.DataFrame([70])
    
    particle_df['n_10_130'] = n_10_130
    
    ##--Compute TOTAL counts from all size bins combined--##
    particle_df['Total_particles_STP'] = (particle_df['n_3_10'].fillna(0) + 
          particle_df['n_10_130'].fillna(0) + particle_df['PCTcon'].fillna(0))
    
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
    
    time_averaged = data['Time']
    
    ##--Calculate time edges for each bin--##
    time_step = time_averaged.iloc[1] - time_averaged.iloc[0]  
    time_edges = np.append(time_averaged, time_averaged.iloc[-1] + time_step)  # length N + 1
    
    ##--Concatenate bin centers and reindex--##
    bin_centers = pd.concat([n_10_130_center, PCASP_bin_center], axis=0).reset_index(drop=True)
    
    ##--Place all binned data in a single df--##
    all_bins_aligned = PCASP_STP
    all_bins_aligned['6.25'] = particle_df['n_3_10']
    all_bins_aligned['70'] = particle_df['n_10_130']
    
    time_index = data['Time']  # use the same index as coagulation_sink
    
    ##--Ensure particle bin dataframes are indexed to time_index and properly named--##
    diameter_dfs = {}
    for col in all_bins_aligned.columns:
        vals = all_bins_aligned[col].to_numpy()
        # explicitly name the column as the bin diameter
        diameter_dfs[col] = pd.DataFrame({str(col): vals}, index=time_index[-len(vals):])
    
    #####################################
    ##--Coagulation sink calculations--##
    #####################################
    
    ##--Constants--##
    R = 8.314  # m^3*Pa*K^-1*mol^-1
    k = 1.38E-23  # m^2*kg*s^-2*K^-1
    C = 1.458E-6  # kg/ms*sqrt(K)
    S = 110.4  # K
    MMair = 28.96  # g/mol
    Mair = MMair/(6.022E23 * 1000)  # kg
    Dair = 3.61E-10  # m
    
    ##--For nucleation particles N(2.5-10)--##
    nuc_diam = 6.25E-9  # m
    nuc_vol = (4/3) * np.pi * (nuc_diam / 2) ** 3
    nuc_mass = nuc_vol
    z_nuc = nuc_mass / Mair
    sigma_nuc = (Dair + nuc_diam) / 2
    
    ##--Canonical time index from CSV--##
    time_index = data['Time']  # use as master index for all series
    
    ##--Ensure particle bin dataframes are indexed to time_index--##
    diameter_dfs = {}
    for col in all_bins_aligned.columns:
        vals = all_bins_aligned[col].to_numpy()
        # broadcast values to last N times of master index
        diameter_dfs[col] = pd.DataFrame({col: vals}, index=time_index[-len(vals):])
    
    ##--Convert temperature and pressure to series with aligned index--##
    temperature_series = pd.Series(temperature.values, index=time_index[-len(temperature):])
    pressure_series = pd.Series(pressure.values, index=time_index[-len(pressure):])
    
    Latitude_series = pd.Series(latitude, index=time_index)
    
    
    ##--Concentration of air molecules (number density)##
    Nair = (6.022E23 * pressure_series) / (R * temperature_series)  # num/m^3
    
    ##--Dynamic viscosity (Sutherland's law)##
    dynam_viscosity = (C * temperature_series ** (3/2)) / (temperature_series + S)
    
    ##--Mean speed of nucleation particles##
    nuc_speed = np.sqrt((8 * k * temperature_series) / (np.pi * nuc_mass))
    
    ##--Mean free path against air for slip correction##
    nuc_mfp_estimate = 1/(np.pi * np.sqrt(1 + z_nuc) * Nair * sigma_nuc**2)
    
    ##--Knudsen number and Cunningham slip correction##
    nuc_knudsen = nuc_mfp_estimate / (nuc_diam/2)
    nuc_slip = 1 + 2 * nuc_knudsen * (2.514 + 0.800 * np.exp(-0.550 / nuc_knudsen))
    
    ##--Diffusivity of nucleation particles##
    nuc_diffusivity = (k * temperature_series * nuc_slip / (3 * np.pi * dynam_viscosity * nuc_diam))
    
    ##--Mean free path of nucleation particles##
    nuc_mfp = (8 * nuc_diffusivity / (np.pi * nuc_speed))
    
    ##--g coefficient for nucleation particles##
    nuc_g = (np.sqrt(2) / (3 * nuc_diam * nuc_mfp)) * ((nuc_diam + nuc_mfp)**3 - (nuc_diam**2 + nuc_mfp**2)**(3/2)) - nuc_diam
    
    ##--Initialize coagulation sink series##
    coagulation_sink = pd.Series(0, index=time_index)
    
    #####################################
    ##--Loop through diameter bins--##
    #####################################
    for diameter, df in diameter_dfs.items():
        
        ##--Convert bin name to float diameter in meters##
        mean_diameter = float(diameter) * 1e-9  # m
    
        ##--Particle volume and mass (density = 1)##
        volume = (4/3) * np.pi * (mean_diameter / 2) ** 3
        mass = volume
    
        ##--Mean particle speed##
        speed = np.sqrt((8 * k * temperature_series) / (np.pi * mass))
    
        ##--Reduced mass ratio##
        z = mass / Mair
    
        ##--Collision cross section with air##
        sigma = (mean_diameter + Dair) / 2
    
        ##--Estimate mean free path against air for slip correction##
        mfp_estimate = 1 / (np.pi * np.sqrt(1 + z) * Nair * sigma**2)
    
        ##--Knudsen number and slip correction##
        knudsen_number = mfp_estimate / (mean_diameter / 2)
        slip = 1 + 2 * knudsen_number * (2.514 + 0.800 * np.exp(-0.550 / knudsen_number))
    
        ##--Particle diffusivity##
        diffusivity = (k * temperature_series * slip / (3 * np.pi * dynam_viscosity * mean_diameter))
    
        ##--Mean free path of H2SO4##
        mean_free_path = (8 * diffusivity / (np.pi * speed))
    
        ##--g coefficient##
        g = (np.sqrt(2) / (3 * mean_diameter * mean_free_path)) * \
            ((mean_diameter + mean_free_path)**3 - (mean_diameter**2 + mean_free_path**2)**(3/2)) - mean_diameter
    
        ##--Particle concentration from bin (converted to #/m^3)##
        df['Particle_concentration'] = df.iloc[:,0].fillna(0) / 1e-6
    
        ##--Coagulation kernel per bin##
        df['Coagulation_kernel'] = (2 * np.pi * (nuc_diffusivity + diffusivity) * (nuc_diam + mean_diameter) * 
            ((nuc_diam + mean_diameter) / (nuc_diam + mean_diameter + 2*np.sqrt(nuc_g**2 + g**2))
             + 8 * (nuc_diffusivity + diffusivity) / np.sqrt(nuc_speed**2 + speed**2) / (nuc_diam + mean_diameter))**-1)
    
        ##--Coagulation per bin##
        df['Coagulation'] = df['Coagulation_kernel'] * df['Particle_concentration']
    
        ##--Add to total coagulation sink (reindex to fill missing times with 0)##
        coagulation_sink += df['Coagulation'].reindex(coagulation_sink.index, fill_value=0)
    
    coagulation_sink = pd.DataFrame({'Coagulation': coagulation_sink})
    
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
    
    particle_df['PTemp'] = potential_temp
    
    ###########################
    ##--Create 2D histogram--##
    ###########################
    
    ##--Float type NaNs in potential_temp cannot convert to int, so must be removed--##
    Coagulation_df = pd.DataFrame({
        'PTemp': np.asarray(potential_temp),
        'Latitude': latitude.to_numpy(),
        'Coagulation': coagulation_sink['Coagulation'].to_numpy()
    }, index=time_index)
    Coagulation_clean_df = Coagulation_df.dropna()
    
    ##--Compute global min/max values across all data BEFORE dropping NaNs--##
    lat_min, lat_max = np.nanmin(latitude), np.nanmax(latitude)
    ptemp_min, ptemp_max = np.nanmin(potential_temp), np.nanmax(potential_temp)
    
    ##--Generate common bin edges using specified number of bins--##
    common_lat_bin_edges = np.linspace(lat_min, lat_max, num_bins_lat + 1)
    common_ptemp_bin_edges = np.linspace(ptemp_min, ptemp_max, num_bins_ptemp + 1)
    
    ##--Make 2D histograms using common bins--##
    Coagulation_bin_medians, _, _, _ = binned_statistic_2d(Coagulation_clean_df['Latitude'], 
        Coagulation_clean_df['PTemp'], Coagulation_clean_df['Coagulation'], statistic='median', 
        bins=[common_lat_bin_edges, common_ptemp_bin_edges])
    
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
    Coagulation_plot = ax1.pcolormesh(common_lat_bin_edges, common_ptemp_bin_edges, Coagulation_bin_medians.T,  # Transpose to align correctly
        shading='auto', cmap=new_cmap, vmin=0, vmax=0.0002)
    
    ##--Add colorbar--##
    cb = fig1.colorbar(Coagulation_plot, ax=ax1)
    cb.minorticks_on()
    cb.ax.tick_params(labelsize=16)
    cb.set_label('N(2.5-10) Coagulation Sink (s-1)', fontsize=16)
    
    ##--Set axis labels--##
    ax1.set_xlabel('Latitude (°)', fontsize=16)
    ax1.set_ylabel('Potential Temperature \u0398 (K)', fontsize=16)
    ax1.tick_params(axis='both', labelsize=16)
    ax1.set_title(f"N(2.5-10) Coagulation Sink - {flight.replace('Flight', 'Flight ')} ({flight_date})", fontsize=18)
    ax1.set_ylim(250, 310)
    ax1.set_xlim(67, 77)
    
    ##--Use f-string to save file with flight# appended--##
    Coagulation_output_path = f"{output_path}\\{flight}"
    plt.savefig(Coagulation_output_path, dpi=600, bbox_inches='tight') 
    
    plt.tight_layout()
    plt.show()
    
    ########################
    ##--Diagnostic Plots--##
    ########################
    
    ##--Remove hashtags below to comment out this section--##
    
    ##--Counts per bin for CPC3 data--##
    Coagulation_bin_counts, _, _, _ = binned_statistic_2d(Coagulation_clean_df['Latitude'], 
        Coagulation_clean_df['PTemp'], Coagulation_clean_df['Coagulation'], statistic='count', bins=[common_lat_bin_edges, common_ptemp_bin_edges])
    
    ##--Particles larger than 3 nm--##
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    
    ##--Make special color map where 0 values are white--##
    new_cmap = plt.get_cmap('inferno')
    ##--Values under specified minimum will be white--##
    new_cmap.set_under('w')
    
    ##--Use pcolormesh for the plot, set minimum value for viridis colors as 1--##
    Coagulation_plot = ax1.pcolormesh(common_lat_bin_edges, common_ptemp_bin_edges, Coagulation_bin_counts.T,  # Transpose to align correctly
        shading='auto', cmap=new_cmap, vmin=1, vmax=25)
    
    ##--Add colorbar--##
    cb = fig1.colorbar(Coagulation_plot, ax=ax1)
    cb.minorticks_on()
    cb.ax.tick_params(labelsize=16)
    cb.set_label('Number of Data Points', fontsize=16)
    
    # Set axis labels
    ax1.set_xlabel('Latitude (°)', fontsize=16)
    ax1.set_ylabel('Potential Temperature \u0398 (K)', fontsize=16)
    ax1.tick_params(axis='both', labelsize=16)
    ax1.set_title(f"Coagulation Sink Counts per Bin - {flight.replace('Flight', 'Flight ')} ({flight_date})", fontsize=18)
    ax1.set_ylim(250, 310)
    ax1.set_xlim(67, 77)
    
    ##--Use f-string to save file with flight# appended--##
    CS10_diag_output_path = f"{output_path}\\{flight}_diagnostic"
    plt.savefig(CS10_diag_output_path, dpi=600, bbox_inches='tight') 
    
    plt.tight_layout()
    plt.show()
    #'''
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 16:03:43 2025

@author: repooley
"""

import icartt
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from datetime import date

###################
##--User inputs--##
###################

##--Set the base directory to project folder--##
directory = r"C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data\raw"

##--Flights to analyze - flights 1-10 (flight 1 has missing data)--##
flights_to_analyze = ["Flight2", "Flight3", "Flight4", "Flight5", "Flight6", 
                      "Flight7", "Flight8", "Flight9", "Flight10"]

##--Assign dates to the flights--##
flight_dates = {"Flight1":  date(2015, 4, 5),
    "Flight2":  date(2015, 4, 7),
    "Flight3":  date(2015, 4, 8),
    "Flight4":  date(2015, 4, 8),
    "Flight5":  date(2015, 4, 9),
    "Flight6":  date(2015, 4, 11),
    "Flight7":  date(2015, 4, 13),
    "Flight8":  date(2015, 4, 20),
    "Flight9":  date(2015, 4, 20),
    "Flight10": date(2015, 4, 21)}

##--Pull datasets with zeros not filtered out--##
CPC3_R1 = icartt.Dataset(r"C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data\raw\CPC_R1\CPC3776_Polar6_20150408_R1_L2.ict")    
CPC10_R1 = icartt.Dataset(r'C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data\raw\CPC_R1\CPC3772_Polar6_20150408_R1_L2.ict')

##--Bin data are in a CSV file--##
UHSAS_bins = pd.read_csv(r"C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data\raw\NETCARE2015_UHSAS_bins.csv")


##--Base output path in directory--##
output_path = r"C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data\processed\PTempBinnedData\Particle"

#%%
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

#%%
#################
##--Pull data--##
#################

##--Store processed data here: --##
unified_dfs = []

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
        print(f"No UHSAS file found for {flight}. Skipping...")
        continue
    
    ##--OPC data--##
    OPC_files = find_files(flight_dir, "OPC")
    if OPC_files:
        OPC = icartt.Dataset(OPC_files[0])
    else: 
        print(f"Missing OPC data for {flight}. Skipping...")
    
    #########################
    ##--Pull & align data--##
    #########################
    
    ##--AIMMS Data--##
    altitude = aimms.data['Alt'] # in m
    latitude = aimms.data['Lat'] # in degrees
    temperature = aimms.data['Temp'] + 273.15 # in K
    pressure = aimms.data['BP'] # in pa
    aimms_time =aimms.data['TimeWave'] # seconds since midnight
    
    ##--Establish AIMMS start/stop times--##
    aimms_end = aimms_time.max()
    aimms_start = aimms_time.min()
     
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
    ##--Total count is computed for N > 85 nm--##
    ICARTT_UHSAS_total = UHSAS.data['total_number_conc'] # particles/cm^3

    ##--Make list of columns to pull, each named bin_x--##
    ##--Bins 1-13 not trustworthy. Bins 76-99 overlap with OPC, discard--##
    ##--Trim to use bins 14-76 (500>85 nm)--##
    UHSAS_bin_num = [f'bin_{i}' for i in range(14, 75)]

    ##--Information for bins 14 thru 99--##
    UHSAS_bin_center = UHSAS_bins['bin_avg'].iloc[14:75]
    UHSAS_lower_bound = UHSAS_bins['lower_bound'].iloc[14:75]
    UHSAS_upper_bound = UHSAS_bins['upper_bound'].iloc[14:75]

    ##--Put column names and content in a dictionary and then convert to a Pandas df--##
    UHSAS_bin_names = pd.DataFrame({col: UHSAS.data[col] for col in UHSAS_bin_num})

    ##--Create new column names by rounding the bin center values to the nearest integer--##
    UHSAS_new_col_names = UHSAS_bin_center.round().astype(int).tolist()

    ##--Rename the UHSAS_bins df columns to bin average values--##
    UHSAS_bin_names.columns = UHSAS_new_col_names
    
    ##--Add time to UHSAS_bins df--##
    UHSAS_bin_names.insert(0, 'Time', UHSAS_time)

    ##--Align UHSAS_bins time to AIMMS time--##
    UHSAS_bins_aligned = UHSAS_bin_names.set_index('Time').reindex(aimms_time)
    
    ##--OPC Data--##
    OPC_time = OPC.data['Time_UTC'] # seconds since midnight
    
    ##--Bin data are in a CSV file--##
    OPC_bin_info = pd.read_csv(r"C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data\raw\NETCARE2015_OPC_bins.csv")
    
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

    ###############################
    ##--De-Normalize UHSAS Data--##
    ###############################

    ##--Calculate dlogDp for UHSAS bins--##
    UHSAS_dlogDp = np.log(UHSAS_upper_bound.values) - np.log(UHSAS_lower_bound.values)

    ##--Get only particle count data (excluding 'Time')--##
    UHSAS_particle_counts = UHSAS_bins_aligned.loc[:, UHSAS_new_col_names]  # Adjust column names as needed

    ##--De-Normalize counts by multiplying by dlogDp across all rows--##
    UHSAS_denorm_counts = UHSAS_particle_counts.multiply(UHSAS_dlogDp, axis=1)
    
    ############################
    ##--Standardize OPC data--##
    ############################
    
    ##--Use the de-normalized values for calculating NPF--##
    
    ##--OPC samples every six seconds. Most rows are NaN--##
    ##--Forward-fill NaN values to propagate last valid reading--##
    ##--Limit forward filling to 5 NaN rows--##
    OPC_bins_filled = OPC_bins_aligned.ffill(limit=5)
    
    ##--Calculate dlogDp for each bin in numpy array--##
    dlogDp = np.log(OPC_upper_bound.values) - np.log(OPC_lower_bound.values)
    
    ##--Get only particle count data (excluding 'Time')--##
    OPC_particle_counts = OPC_bins_filled.loc[:, OPC_new_col_names]
    
    ##--Convert to STP!--##
    P_STP = 101325  # Pa
    T_STP = 273.15  # K
    
    ##--DENORM OPC--##
    OPC_conc_STP = []
    
    for OPC, T, P in zip(OPC_particle_counts.values, temperature, pressure):
        if np.isnan(T) or np.isnan(P):
            ##--Append with NaN if any input is NaN--##
            OPC_conc_STP.append([np.nan]*len(OPC))
        else:
            ##--Perform conversion if all inputs are valid--##
            corrected_OPC = OPC * (P_STP / P) * (T / T_STP)
            OPC_conc_STP.append(corrected_OPC)
            
    ##--Convert back to DataFrame with same columns and index--##
    OPC_conc_STP = pd.DataFrame(OPC_conc_STP, columns=OPC_particle_counts.columns, index=OPC_particle_counts.index)
    
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

    ######################
    ##--Calc N(2.5-10)--##
    ######################

    ##--Convert to STP--##
    P_STP = 101325  # Pa
    T_STP = 273.15  # K

    ##--Create empty list for CPC3 particles--##
    CPC3_conc_STP = []

    for CPC3, T, P in zip(CPC3_conc_aligned, temperature, pressure):
        if np.isnan(CPC3) or np.isnan(T) or np.isnan(P):
            ##--Append with NaN if any input is NaN--##
            CPC3_conc_STP.append(np.nan)
        else:
            ##--Perform conversion if all inputs are valid--##
            CPC3_conversion = CPC3 * (P_STP / P) * (T / T_STP)
            CPC3_conc_STP.append(CPC3_conversion)
        
    ##--Create empty list for CPC10 particles--##
    CPC10_conc_STP = []

    for CPC10, T, P in zip(CPC10_conc_aligned, temperature, pressure):
        if np.isnan(CPC10) or np.isnan(T) or np.isnan(P):
            ##--Append with NaN if any input is NaN--##
            CPC10_conc_STP.append(np.nan)
        else:
            ##--Perform conversion if all inputs are valid--##
            CPC10_conversion = CPC10 * (P_STP / P) * (T / T_STP)
            CPC10_conc_STP.append(CPC10_conversion)

    ##--Creates a Pandas dataframe for particle data--##
    df = pd.DataFrame({'PTemp': potential_temp, 
                       'CPC3_conc':CPC3_conc_STP, 'CPC10_conc': CPC10_conc_STP})

    ##--Calculate N3-10 particles--##
    nuc_particles = (df['CPC3_conc'] - df['CPC10_conc'])

    ##--Change calculated particle counts less than zero to NaN--##
    nuc_particles = np.where(nuc_particles >= 0, nuc_particles, np.nan)

    ##--Add nucleating particles to df--##
    df['nuc_particles'] = nuc_particles
    
    #####################
    ##--Calc N(10-89)--##
    #####################

    ##--Re-compute UHSAS total count using denormalized data--##
    UHSAS_total = UHSAS_denorm_counts.sum(axis=1)

    ##--Create df with UHSAS total counts and index to AIMMS time--##
    UHSAS_total_aligned = pd.DataFrame({'Time': aimms_time, 'Total_count': UHSAS_total}).set_index('Time')
    
    ##--Same for OPC--##
    OPC_total = OPC_conc_STP.sum(axis=1)
    
    OPC_total_aligned = pd.DataFrame({'Time': aimms_time, 'Total_count': OPC_total}).set_index('Time')

    ##--Create df with CPC10 counts and set index to time--##
    CPC10_counts = pd.DataFrame({'Time':aimms_time, 'Counts':CPC10_conc_STP}).set_index('Time')

    ##--Calculate particles below UHSAS lower cutoff--##
    n_10_89 = (CPC10_counts['Counts'] - (UHSAS_total_aligned['Total_count'] + OPC_total_aligned['Total_count']))

    ##--Change calculated particle counts less than zero to NaN--##
    n_10_89 = np.where(n_10_89 >= 0, n_10_89, np.nan)

    ##--Add 10-89 nm particles to the dataframe--##
    df['n_10_89'] = n_10_89
        
    #############################
    ##--Calculate Uncertainty--##        
    #############################

    ##--Pull CPC data from R1 data--##
    CPC3_R1_conc = CPC3_R1.data['conc']
    CPC10_R1_conc = CPC10_R1.data['conc']

    ##--Isolate zero periods, setting conservative upper limit of 50 counts--##
    ##--Numpy doesn't recognize -9999 as NaN, tell it to ignore these values--##
    CPC3_zeros_c = CPC3_R1_conc[(CPC3_R1_conc < 50) & (CPC3_R1_conc != -9999)]
    CPC10_zeros_c = CPC10_R1_conc[(CPC10_R1_conc < 50) & (CPC10_R1_conc != -99999)]

    ##--Calculate standard deviation of zeros--##
    CPC3_sigma = np.std(CPC3_zeros_c, ddof=1)  # Use ddof=1 for sample standard deviation
    CPC10_sigma = np.std(CPC10_zeros_c, ddof=1)

    ##--UHSAS doesn't have zero periods, using Poisson counting uncertainty--##
    UHSAS_total_sqrt = np.sqrt(UHSAS_denorm_counts)

    ##--Use simple sum of UHSAS uncertainties per bin for conservative estimate--##
    ##--Similar result as using sqrt of squares but erring on side of caution--##
    UHSAS_total_error = UHSAS_total_sqrt.sum(axis=1)
    
    ##--Repeat for OPC data--##
    OPC_total_sqrt = np.sqrt(OPC_conc_STP)
    
    OPC_total_error = OPC_total_sqrt.sum(axis=1)

    # %%
    #############################
    ##--Propagate uncertainty--##
    #############################

    ##--The ICARTT files for CPC instruments say 10% uncertainty of meas value - feels conservative for large counts!--##
    ##--Calculate the 3 sigma uncertainty for nucleating particles--##

    T_error = 0.3 # K, constant
    P_error = 100 + 0.0005*(pressure)

    ##--Use formula for mult/div to compute error after converting to STP--##
    greater3nm_error = (CPC3_conc_aligned)*(((P_error)/(pressure))**2 + ((T_error)/(temperature))**2 + ((CPC3_sigma)/(CPC3_conc_aligned)))**(0.5)
    greater10nm_error = (CPC10_conc_aligned)*(((P_error)/(pressure))**2 + ((T_error)/(temperature))**2 + ((CPC10_sigma)/(CPC10_conc_aligned)))**(0.5)

    ##--Use add/subtract forumula to compute 3sigma error--##
    nuc_error_3sigma = (((greater3nm_error)**2 + (greater10nm_error)**2)**(0.5))*3

    ##--nuc_error_3sigma still has a time index, reset to integer to align--##
    df['nuc_error_3sigma'] = nuc_error_3sigma

    ##--Calculate error in difference between CPC10 and UHSAS + OPC--##
    aitken_error_3sigma = (((greater10nm_error)**2 + (UHSAS_total_error)**2 + (OPC_total_error)**2)**(0.5))*3
        
    ##--Add uncertainty for 10-85 nm bin to big df--##
    df['aitken_error_3sigma'] = aitken_error_3sigma
    
    #########################
    ##--Create dataframes--##
    #########################

    ##--Convert everything to a single DataFrame--##
    df = pd.DataFrame({
        'Ptemp': potential_temp,
        'Latitude': latitude,
        'CPC3_conc': CPC3_conc_STP,
        'CPC10_conc': CPC10_conc_STP,
        'nuc_particles': nuc_particles,
        'grow_particles': n_10_89,
        'nuc_error': nuc_error_3sigma,
        'grow_error': aitken_error_3sigma
    })
    
    ##--Drop nans--##
    df = df.dropna(subset=[
        'Ptemp', 'Latitude',
        'CPC3_conc', 'CPC10_conc',
        'nuc_particles', 'grow_particles'
    ]).reset_index(drop=True)
    
    ##--Store for later binning--##
    unified_dfs.append(df)

#%%
###############
##--BINNING--##
###############

nuc_uncertainties = []
grow_uncertainties = []
num_bins = 60

for df in unified_dfs:

    ##--Bin edges--##
    bin_edges = np.linspace(df['Ptemp'].min(), df['Ptemp'].max(), num_bins + 1)

    ##--Cut into Ptemp bins--##
    df['PTemp_bin'] = pd.cut(df['Ptemp'], bins=bin_edges)

    ##--Compute bin medians, including uncertainty--##
    binned_df = df.groupby('PTemp_bin', observed=False).agg(
        PTemp_center=('Ptemp', 'median'),
        CPC10_conc_center=('CPC10_conc', 'median'),
        CPC3_conc_center=('CPC3_conc', 'median'),
        nuc_particles_center=('nuc_particles', 'median'),
        nuc_error_median=('nuc_error', 'median'),
        grow_particles_center=('grow_particles', 'median'),
        grow_error_median=('grow_error', 'median')
    ).reset_index()

    ##--Store flight-level mean-median uncertainties--##
    nuc_uncertainties.append(binned_df['nuc_error_median'].mean())
    grow_uncertainties.append(binned_df['grow_error_median'].mean())

#%%

################
##--PLOTTING--##
################

fig, axs = plt.subplots(1, 4, figsize=(12, 6), sharey=True)

##--Colormap - assign a color to each flight--##
cmap = plt.cm.viridis
n_flights = len(unified_dfs)
colors = [cmap(i / (n_flights - 1)) for i in range(n_flights)]

##--Loop over flights--##
for i, (flight, df) in enumerate(zip(flights_to_analyze, unified_dfs)):
    
    ####################################
    ##--Assign date to flight number--##
    ####################################

    flight_date = flight_dates[flight]  

    ##--Bin edges--##
    min_ptemp = df["Ptemp"].min()
    max_ptemp = df["Ptemp"].max()
    bin_edges = np.linspace(min_ptemp, max_ptemp, num_bins + 1)
    
    ##--Potential temperature bins--##
    df["PTemp_bin"] = pd.cut(df["Ptemp"], bins=bin_edges)
    
    ##--Bin medians--##
    binned_df = df.groupby("PTemp_bin", observed=False).agg(
        PTemp_center=("Ptemp", "median"),
        CPC10_conc_center=("CPC10_conc", "median"),
        CPC3_conc_center=("CPC3_conc", "median"),
        nuc_particles_center=("nuc_particles", "median"),
        grow_particles_center=("grow_particles", "median")
    ).reset_index()
    
    axs[0].plot(binned_df["CPC3_conc_center"], binned_df["PTemp_center"],
                color=colors[i], label=f'Flight {i+1} ({flight_date})')
    
    axs[1].plot(binned_df["CPC10_conc_center"], binned_df["PTemp_center"],
                color=colors[i], label=f'Flight {i+1} ({flight_date})')
    
    axs[2].plot(binned_df["nuc_particles_center"], binned_df["PTemp_center"],
                color=colors[i], label=f'Flight {i+1} ({flight_date})')
    
    axs[3].plot(binned_df["grow_particles_center"], binned_df["PTemp_center"],
                color=colors[i], label=f'Flight {i+1} ({flight_date})')

##--Subplot 1--##
axs[0].set_ylabel("Potential Temperature (K)", fontsize=16)
axs[0].set_xlabel("Counts/cm³", fontsize=14)
axs[0].set_title("N ≥ 10 nm", fontsize=16)
axs[0].set_xlim(-50, 2000)
axs[0].tick_params(axis='both', labelsize=12)
axs[0].axhline(y=285, color="k", linestyle="--", linewidth=1)
axs[0].axhline(y=299, color="k", linestyle="--", linewidth=1)

##=-Polar dome labels--##
x_text = axs[0].get_xlim()[0] + 1050
axs[0].text(x_text, 282, "Polar Dome", fontsize=11, color="k",
            verticalalignment="center", horizontalalignment="left")
axs[0].text(x_text, 288, "Marginal Dome", fontsize=11, color="k",
            verticalalignment="center", horizontalalignment="left")

##--Subplot 2--##
axs[1].set_title("N ≥ 2.5 nm", fontsize=16)
axs[1].set_xlabel("Counts/cm³", fontsize=14)
axs[1].set_xlim(-50, 3400)
axs[1].tick_params(axis='both', labelsize=12)
axs[1].axhline(y=285, color="k", linestyle="--", linewidth=1)
axs[1].axhline(y=299, color="k", linestyle="--", linewidth=1)

##--Subplot 3--##
axs[2].set_title("$N_{2.5-10}$", fontsize=16)
axs[2].set_xlabel("Counts/cm³", fontsize=14)
axs[2].tick_params(axis='both', labelsize=12)
axs[2].axhline(y=285, color="k", linestyle="--", linewidth=1)
axs[2].axhline(y=299, color="k", linestyle="--", linewidth=1)

##--Add mean uncertainty--##
global_uncertainty_line = np.mean(nuc_uncertainties)
axs[2].axvline(global_uncertainty_line, color='crimson', linestyle='dashed', linewidth=1, label='3$\sigma$ Uncertainty')

##--Subplot 4--##
axs[3].set_title("$N_{10-89}$", fontsize=16)
axs[3].set_xlabel("Counts/cm³", fontsize=14)
axs[3].tick_params(axis='both', labelsize=12)
axs[3].axhline(y=285, color="k", linestyle="--", linewidth=1)
axs[3].axhline(y=299, color="k", linestyle="--", linewidth=1)

global_uncertainty_line2 = np.mean(grow_uncertainties)
axs[3].axvline(global_uncertainty_line2, color='crimson', linestyle='dashed', linewidth=1, label='3$\sigma$ Uncertainty')


axs[3].legend(loc="lower right", fontsize=10)

plt.suptitle("NETCARE 2015 Vertical Particle Profiles", fontsize=18)

plt.tight_layout(rect=[0, 0.05, 1, 0.99])
plt.show()

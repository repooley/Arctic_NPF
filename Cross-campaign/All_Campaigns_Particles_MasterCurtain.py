# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 11:19:45 2026

@author: repooley
"""

import sys
import icartt
import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic_2d
import matplotlib.pyplot as plt 
import cmcrameri as cm
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import string
from pathlib import Path
##--Import modules from NETCARE utils folder--##
from NETCARE_loader import load_flight # loads and aligns data
from Particle_bin_calculator import calc_particle_bins 

###########################
##--Establish directory--##
###########################

##--Path to this script--##
script_path = Path(__file__).resolve()

##--Path to the root which is 1 level up in the directory--##
root = script_path.parents[1]

##--Paths to raw data--##
ATom_directory = root / "ATom2018" / "data" / "raw"
NETCARE_directory = root / "NETCARE2015" / "data" / "raw"
PAMARCMiP_directory = root / "PAMARCMiP2012" / "data" / "raw"
FIREACE_directory = root / "FIREACE1998" / "data" / "raw"

##--Path to utils folder containing NETCARE alignment + calc scripts--##
#sys.path.insert(0, str(root / "NETCARE2015" / "src" / "utils"))

##--Base output path for figures in directory--##
output_path = root / "Cross-campaign" / "data" / "processed"

##--Make the output path directory if it doesn't already exist--##
os.makedirs(output_path, exist_ok=True)

#####################################
##--Binned instrument information--##
#####################################

##--UHSAS--##

##--Bin data are in a CSV file--##
##--SAME bins for NETCARE and PAMARCMiP--##
UHSAS_bins = pd.read_csv(r"C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data\raw\NETCARE2015_UHSAS_bins.csv")

##--Make list of columns to pull, each named bin_x--##
UHSAS_bin_num = [f'UH_bin_{i}' for i in range(1, 100)]

##--Information for bins 1 thru 99--##
UHSAS_bin_center = UHSAS_bins['bin_avg'].iloc[0:100]
UHSAS_lower_bound = UHSAS_bins['lower_bound'].iloc[0:100]
UHSAS_upper_bound = UHSAS_bins['upper_bound'].iloc[0:100]

##--PCASP--##

##--Path to csv file containing PCASP bin info--##
PCASP_bins_path = r"C:\Users\repooley\REP_PhD\Arctic_NPF\FIREACE1998\data\raw\FIREACE1998_PCASP_bins.csv"

PCASP_bins = pd.read_csv(PCASP_bins_path)

##--15 total bins--##
PCASP_bin_num = [f'bin_{i}' for i in range(1, 16)]

##--Information for bins--##
PCASP_bin_center = PCASP_bins['bin_avg']
PCASP_lower_bound = PCASP_bins['lower_bound']
PCASP_upper_bound = PCASP_bins['upper_bound']

############################
##--Analysis constraints--##
############################

##--Choose which flights to analyze here!--##
##--ATom--##
ATom_to_analyze = ["Flight2", "Flight10", "Flight11", "Flight12"]

##--NETCARE: Flights 1-10--##
##--Flight 1 no UHSAS data, Flight 4 bad CPC data--##
NETCARE_to_analyze = ["Flight2", "Flight3", "Flight5", "Flight6", 
                      'Flight7', 'Flight8', 'Flight9', 'Flight10']

##--PAMARCMiP: Flights 1-9--##
##--Flight 5 bad data--##
PAMARCMiP_to_analyze = ["Flight1", "Flight2", "Flight3", "Flight6", 
                      "Flight7", "Flight8", "Flight9"]

##--FIRE-ACE: Flights 1-18--##
##--Flights 1, 2, 4, 5, & 6 missing 1hz or 2min averaged data--##
FIREACE_to_analyze = ["Flight3",  "Flight7", "Flight8", "Flight9", "Flight10", 
                      "Flight11", "Flight12", "Flight13", "Flight14", 
                      "Flight15", "Flight16", "Flight17", "Flight18"]

##--Set number of bins for latitude and potential temperature--##
num_bins_lat = 12
num_bins_ptemp = 12

#######################
##--Calculate PTemp--##
#######################

##--Constants--##
p_0 = 1E5 # Reference pressure in Pa (1000 hPa)
k = 0.286 # Poisson constant for dry air

##--Calculation as a function--##
def calc_ptemp(temperature, pressure):
    
    potential_temp = []
    
    for T, P in zip(temperature, pressure):
        p_t = T*(p_0/P)**k
        potential_temp.append(p_t)
        
    return potential_temp

##################
##--Open Files--##
##################

##--Define function that creates datasets from filenames--##
def find_files(directory, flight, partial_name):
    ##--flight data are stored in a folder called "raw"--##
    flight_dir = os.path.join(directory, flight)
    search_pattern = os.path.join(flight_dir, f"*{partial_name}*")
    return sorted(glob.glob(search_pattern))


##--ATom--##
ATom_nucleating_dfs = []
ATom_aitken_dfs = []
ATom_diameter_dfs = []
ATom_conditions_dfs = []

##--Loop through each flight in the list--##
for flight in ATom_to_analyze:
    
    #########################
    ##--Open ICARTT Files--##
    #########################

    ##--Pull the merged datasets--##
    ATom_dataset = icartt.Dataset(find_files(ATom_directory, flight, "MER")[0])

    #################
    ##--Pull data--##
    #################

    ATom_altitude = ATom_dataset.data['G_ALT'] # in m (not sure if this is best one)
    ATom_latitude = ATom_dataset.data['LAT_AMSSD'] # deg
    ATom_temperature = ATom_dataset.data['T'] # in K
    ATom_pressure = ATom_dataset.data['P'] * 100 # in Pa
    ATom_RH = ATom_dataset.data['Relative_Humidity'] # wrt water, percent
    ATom_time =ATom_dataset.data['UTC_Start'] # seconds since midnight UTC
    ATom_nucleating = ATom_dataset.data['N_nucl_AMP'] # num/cm^3 STP (2.7-12 nm)
    ATom_aitken = ATom_dataset.data['N_aitken_AMP'] # num/cm^3 STP (12-60 nm)
    ATom_accumulation = ATom_dataset.data['N_accum_AMP'] # num/cm^3 STP (60 nm - 0.5 um)
    ATom_coarse = ATom_dataset.data['N_coarse_AMP'] # num/cm^3 STP (0.5 um - 4.8 um)
    
    ##--Constrain latitude to the Arctic region--##
    ATom_latitude[ATom_latitude < 66.5] = np.nan
    
    ##--There are notable outliers in the nucleating data--##
    
    ##--First convert to a series for calc--##
    ATom_nucleating_series = pd.Series(ATom_nucleating)
    
    ##--REMOVE OUTLIERS above 99th percentile--##
    p = 0.99
    
    ##--Compute threshold for each UHSAS column--##
    ATom_nucleating_thresh = ATom_nucleating_series.quantile(p)
    
    ##--keep only rows where each bin is below its threshold--##
    ATom_nucleating_filtered = ATom_nucleating_series[ATom_nucleating_series.le(ATom_nucleating_thresh)]
    
    ATom_nucleating_filtered = ATom_nucleating_series.mask(
    ATom_nucleating_series <= ATom_nucleating_thresh)
    
    ##--Calculate potential temperature using above function--##
    ATom_potential_temp = calc_ptemp(ATom_temperature, ATom_pressure)
        
    ##--Convert ptemp to np array--##
    ATom_potential_temp = np.array(ATom_potential_temp)
        
    ##--Constrain ptemp to range of other three campaigns--##
    ATom_potential_temp[ATom_potential_temp > 310] = np.nan
        
    ##--Append Nucleating and Aitken lists with dataframes--##
    ##--Convert nucleating data to dataframe--##
    ATom_nucleating_df = pd.DataFrame({'nucleating': ATom_nucleating_series, 
                                  'latitude': ATom_latitude, 
                                  'PTemp': ATom_potential_temp, 
                                  'time': ATom_time}).set_index(ATom_time)
    
    ##--Append list of dataframes with ATom nucleating data--##
    ATom_nucleating_dfs.append(ATom_nucleating_df)
    
    ##--Convert aitken mode data to dataframe--##
    ATom_aitken_df = pd.DataFrame({'aitken': ATom_aitken,
                              'latitude': ATom_latitude, 
                              'PTemp': ATom_potential_temp, 
                              'time': ATom_time}).set_index(ATom_time)
    
    ##--Append list of dataframes with ATom aitken data--##
    ATom_aitken_dfs.append(ATom_aitken_df)
        
    ###########################
    ##--Wrangle binned data--##
    ###########################
    
    ##--Concatenate bin edges--##
    ATom_combined_bin_edges = np.concatenate([
        [12],       # upper edge of N(2.7-12), also lower of next
        [60],       # upper edge of N(12-60), also lower of next
        [500],      # upper edge of N(60-500), also lower of next
        [4800]      # upper edge of final bin
        ])
    
    ##--Concatenate bin centers--##
    ATom_bin_centers = np.concatenate([
        [36], 
        [280],
        [2650]
        ])
    
    ##--Place all binned data in a single df--##
    ATom_all_bins_aligned = pd.concat([pd.DataFrame({'36':ATom_aitken, '280':ATom_accumulation, 
                                        '2650':ATom_coarse}, index=ATom_time)], axis=1)
    
    ##--Sum across all bins to get total particle count--##
    ATom_total_particle_count = ATom_all_bins_aligned.sum(axis=1, numeric_only=True) 
    
    ##--Create a dictionary to store each column as a separate dataframe, col names are keys--##
    ATom_diameter_dict = {col: pd.DataFrame({col: ATom_all_bins_aligned[col]}) for col in ATom_all_bins_aligned.columns}
    
    ##--Append list with diameters--##
    ATom_diameter_dfs.append(ATom_diameter_dict)
    
    ##--Create separate conditions df and append existing list--##
    ATom_conditions_dfs.append(pd.DataFrame({'temperature': ATom_temperature, 
                    'pressure': ATom_pressure, 'latitude': ATom_latitude}, index=ATom_time))
    
##--NETCARE--##
NETCARE_nucleating_dfs = []
NETCARE_aitken_dfs = []
NETCARE_diameter_dfs = []
NETCARE_conditions_dfs = []
 
##--Loop through each flight, pulling and analyzing data--##
for flight in NETCARE_to_analyze:
    ##--Follow which flight is processing--##
    print(f"Processing {flight}...")
    
    NETCARE_data = load_flight(NETCARE_directory, flight)
    
    ##--AIMMS data--##
    AIMMS = NETCARE_data["AIMMS"]
    
    NETCARE_time = AIMMS.data["TimeWave"]
    NETCARE_altitude = AIMMS.data["Alt"] # m
    NETCARE_latitude = AIMMS.data['Lat'] # deg
    NETCARE_temperature = AIMMS.data["Temp"] + 273.15 # K
    NETCARE_pressure = AIMMS.data["BP"] # pa

    ##--Pull in the df with particles--##
    Calc_NETCARE_particle_df = calc_particle_bins(NETCARE_data)
     
    NETCARE_particle_df = Calc_NETCARE_particle_df['df']
    
    ##--Calculate potential temperature using the above function--##
    NETCARE_potential_temp = calc_ptemp(NETCARE_temperature, NETCARE_pressure)
    
    ##--Create dataframe for nucleation mode data--##
    NETCARE_nucleating_df = pd.DataFrame({'nucleating': NETCARE_particle_df['nuc_particles'], 
                                  'latitude': NETCARE_latitude, 
                                  'PTemp': NETCARE_potential_temp, 
                                  'time': NETCARE_time}).set_index('time')
    
    ##--Append list of dataframes with ATom nucleating data--##
    NETCARE_nucleating_dfs.append(NETCARE_nucleating_df)
    
    ##--Convert aitken mode data to dataframe--##
    NETCARE_aitken_df = pd.DataFrame({'aitken': NETCARE_particle_df['n_10_89'],
                              'latitude': NETCARE_latitude, 
                              'PTemp': NETCARE_potential_temp, 
                              'time': NETCARE_time}).set_index('time')
    
    ##--Append list of dataframes with ATom aitken data--##
    NETCARE_aitken_dfs.append(NETCARE_aitken_df)
    
    ##--Pull in df with NETCARE bins--##
    NETCARE_diameters = Calc_NETCARE_particle_df['diameter_dfs']
    
    ##--Create a list of all the diameter dfs--##
    NETCARE_diameter_dfs.append(NETCARE_diameters)
    
    ##--Place conditions in a separate df--##
    NETCARE_conditions_dfs.append(pd.DataFrame({'temperature': NETCARE_temperature, 
                'pressure': NETCARE_pressure, 'latitude': NETCARE_latitude}, index=NETCARE_time))

'''    
##--Store processed data here: --##
PAMARCMiP_nucleating_dfs = [] # will be EMPTY - no data
PAMARCMiP_aitken_dfs = []
PAMARCMiP_diameter_dfs = []
PAMARCMiP_conditions_dfs = []
 
##--Loop through each flight, pulling and analyzing data--##
for flight in PAMARCMiP_to_analyze:
    ##--Follow which flight is processing--##
    print(f"Processing {flight}...")
    
    ##--Pull file--##
    PAMARCMiP_data = pd.read_csv(find_files(PAMARCMiP_directory, flight, ".csv")[0])
    
    #################
    ##--Pull data--##
    #################
    
    ##--Data--##
    PAMARCMiP_altitude =  PAMARCMiP_data['Altitude'] # in m
    PAMARCMiP_latitude =  PAMARCMiP_data['Latitude'] # in degrees
    PAMARCMiP_temperature =  PAMARCMiP_data['Temp'] + 273.15 # in K
    PAMARCMiP_pressure =  PAMARCMiP_data['Pressure'] # in pa
    PAMARCMiP_time =  PAMARCMiP_data['Time'] # seconds since midnight
    
    ##--The first datapoint in 'latitude' column is erraneous (47.12 N)--##
    ##--Constrain latitude to the Arctic region--##
    PAMARCMiP_latitude =  PAMARCMiP_latitude.where( PAMARCMiP_latitude >= 66.5, np.nan)
    
    ##--USHAS Data--##
    PAMARCMiP_UHSAS_total_num =  PAMARCMiP_data['UH-TotConc'] # particles/cm^3
    
    ##--10 nm CPC data--##
    PAMARCMiP_CPC10_conc = PAMARCMiP_data['CPC10'] # count/cm^3
    
    ##--Put column names and content in a dictionary and then convert to a Pandas df--##
    UHSAS_bins = pd.DataFrame({col: PAMARCMiP_data[col] for col in UHSAS_bin_num})
    
    ##--Create new column names by rounding the bin center values to the nearest integer--##
    UHSAS_new_col_names = UHSAS_bin_center.round().astype(int).tolist()
    
    ##--Rename the UHSAS_bins df columns to bin average values--##
    UHSAS_bins.columns = UHSAS_new_col_names
    
    ##--REMOVE OUTLIERS above 99th percentile--##
    p = 0.99
    
    ##--Compute threshold for each UHSAS column--##
    PAMARCMiP_UHSAS_thresh = UHSAS_bins.quantile(p)
    
    ##--keep only rows where each bin is below its threshold--##
    PAMARCMiP_UHSAS_bins_filtered = UHSAS_bins[UHSAS_bins.le(
        PAMARCMiP_UHSAS_thresh, axis=1)]
    
    PAMARCMiP_CPC10_thresh = PAMARCMiP_CPC10_conc.quantile(p)
    PAMARCMiP_CPC10_filtered = PAMARCMiP_CPC10_conc[PAMARCMiP_CPC10_conc 
        <= PAMARCMiP_CPC10_thresh]
    
    ###############################
    ##--De-Normalize UHSAS Data--##
    ###############################
    
    ##--For total count calculation--##
    
    ##--Calculate dlogDp for UHSAS bins--##
    UHSAS_dlogDp = np.log(UHSAS_upper_bound.values) - np.log(UHSAS_lower_bound.values)
    
    ##--Get only particle count data (excluding 'Time')--##
    UHSAS_particle_counts = UHSAS_bins.loc[:, UHSAS_new_col_names]  
    
    ##--De-Normalize counts by multiplying by dlogDp across all rows--##
    UHSAS_denorm_counts = UHSAS_particle_counts.multiply(UHSAS_dlogDp, axis=1)

    #####################
    ##--Calc N(10-60)--##
    #####################
    
    ##--Create df with UHSAS total counts--##
    PAMARCMiP_UHSAS_total = pd.DataFrame({'Time': PAMARCMiP_time, 
                                'Total_count': PAMARCMiP_UHSAS_total_num})
    
    ##--Create df with CPC10 counts and set index to time--##
    PAMARCMiP_CPC10_counts = pd.DataFrame({'Time':PAMARCMiP_time, 
                                 'Counts':PAMARCMiP_CPC10_filtered})
    
    ##--Calculate particles below UHSAS lower cutoff--##
    PAMARCMiP_n_10_60 = (PAMARCMiP_CPC10_counts['Counts'] - 
               PAMARCMiP_UHSAS_total['Total_count'])
    
    ##--Change calculated particle counts less than zero to NaN--##
    PAMARCMiP_n_10_60 = np.where(PAMARCMiP_n_10_60 >= 0, PAMARCMiP_n_10_60, np.nan)
    
    ##--Put N(10-60) bin center in a df--##
    PAMARCMiP_n_10_60_center = pd.DataFrame([35])
    
    ##--Flatten--##
    PAMARCMiP_n_10_60_center = pd.Series(PAMARCMiP_n_10_60_center.values.flatten())
    
    ##--Convert n_10_60 to a df--##
    PAMARCMiP_n_10_60_df = pd.DataFrame({'35': PAMARCMiP_n_10_60, 
                                         'time':PAMARCMiP_time})
    
    ##--Calculate potential temperature--##
    PAMARCMiP_ptemp = calc_ptemp(PAMARCMiP_temperature, PAMARCMiP_pressure)
              
    ##--Append Nucleating and Aitken lists with dataframes--##
    PAMARCMiP_nucleating_df = pd.DataFrame({'nucleating': np.full(len(PAMARCMiP_time), 
                                                        np.nan),
                              'latitude': PAMARCMiP_latitude, 
                              'PTemp': PAMARCMiP_ptemp, 
                              'time': PAMARCMiP_time}).set_index(PAMARCMiP_time)
    
    ##--Append list outside of loop with nucleating dfs--##
    PAMARCMiP_nucleating_dfs.append(PAMARCMiP_nucleating_df)
    
    ##--Convert aitken mode data to dataframe--##
    PAMARCMiP_aitken_df = pd.DataFrame({'aitken': PAMARCMiP_n_10_60_df['35'],
                              'latitude': PAMARCMiP_latitude, 
                              'PTemp': PAMARCMiP_ptemp, 
                              'time': PAMARCMiP_time}).set_index(PAMARCMiP_time)
    
    ##--Append list of dataframes with PAMARCMiP aitken data--##
    PAMARCMiP_aitken_dfs.append(PAMARCMiP_aitken_df)
    
    ###########################
    ##--Wrangle binned data--##
    ###########################
    
    ##--Concatenate bin edges--##
    PAMARCMiP_combined_bin_edges = np.concatenate([
        [10],       # lower edge of N(10-60)
        [60],       # upper edge of N(10-60), also lower of next
        UHSAS_upper_bound.values,  # UHSAS bins continue from 60
    ])
    
    ##--Concatenate bin centers and reindex--##
    PAMARCMiP_bin_centers = pd.concat([PAMARCMiP_n_10_60_center, UHSAS_bin_center], 
                            axis=0).reset_index(drop=True)
    
    ##--Place all binned data in a single df--##
    PAMARCMiP_all_bins_aligned = pd.concat([PAMARCMiP_n_10_60_df['35'], 
                                            PAMARCMiP_UHSAS_bins_filtered], axis=1)
    
    ##--Sum across all particle counts to get total--##
    PAMARCMiP_total_particle_count = PAMARCMiP_all_bins_aligned.sum(axis=1, 
                                                        numeric_only=True) 
    
    ##--Create a dictionary to store each column as a separate dataframe, col names are keys--##
    PAMARCMiP_diameter_df = {col: pd.DataFrame({col: 
        PAMARCMiP_all_bins_aligned[col]}) for col in 
        PAMARCMiP_all_bins_aligned.columns}
    
    ##--Append list of diameter dataframes for PAMARCMiP--##
    PAMARCMiP_diameter_dfs.append(PAMARCMiP_diameter_df)
    
    ##--Append list of conditions--##
    ##--Place conditions in a separate df--##
    PAMARCMiP_conditions_dfs.append(pd.DataFrame({'temperature': PAMARCMiP_temperature, 
                            'pressure': PAMARCMiP_pressure, 
                            'latitude': PAMARCMiP_latitude}))
'''
    
FIREACE_nucleating_dfs = []
FIREACE_aitken_dfs = []
FIREACE_diameter_dfs = []
FIREACE_conditions_dfs = []

for flight in FIREACE_to_analyze: 
    
    ##--Pull csv file containing all data--##
    FIREACE_files = find_files(FIREACE_directory, flight, "FIREACE")

    ##--The averaged data is always the second file--##
    if FIREACE_files:
        FIREACE_data = pd.read_csv(FIREACE_files[1])
        ##--1 hz is always first--##
        FIREACE_nonav = pd.read_csv(FIREACE_files[0])

    ##--Pull data variables from averaged file--##
    FIREACE_time = FIREACE_data['Time'] # HHMMSS UTC time
    FIREACE_pressure = FIREACE_data['Pressure'] * 100 # in Pa
    FIREACE_temperature = FIREACE_data['Temperature'] + 273.15 # in K
    FIREACE_altitude = FIREACE_data['Altitude'] # in m (agl?)
    FIREACE_latitude = FIREACE_data['Latitude'] # degrees
    
    ##--Also pull for nonaveraged file--##
    FIREACE_time_nonav = FIREACE_nonav['Time'] # HHMMSS UTC
    FIREACE_pressure_nonav = FIREACE_nonav['Pressure'] * 100 # in Pa
    FIREACE_temperature_nonav = FIREACE_nonav['Temperature'] + 273.15 # in K
    FIREACE_altitude_nonav = FIREACE_nonav['Altitude'] # in m
    FIREACE_latitude_nonav = FIREACE_nonav['Latitude'] # degrees
    
    ##--Constrain latitude to the Arctic region--##
    FIREACE_latitude = FIREACE_latitude.where(FIREACE_latitude >= 66.5, np.nan)

    ##--Particle data, 3 and 10 nm cutoffs, respectively--##
    FIREACE_CPC3_data = FIREACE_data['CN3025'] # Uncorrected data has a flow issue - but corrected not populated for many flights
    FIREACE_CPC10_data = FIREACE_data['CN7610']
    
    ##--Repeat for non-averaged data--##
    FIREACE_CPC3_nonav = FIREACE_nonav['CN3025'] # Uncorrected data has a flow issue - but corrected not populated for many flights
    FIREACE_CPC10_nonav = FIREACE_nonav['CN7610']
    
    ##--PCASP data--##
    FIREACE_PCASP_data = FIREACE_data.iloc[:, 14:29] 

    ##--Add time to PCASP_data--##
    FIREACE_PCASP_data.insert(0, 'Time', FIREACE_data['Time'])

    ##--Set time as the index for later alignment--##
    FIREACE_PCASP_data = FIREACE_PCASP_data.set_index('Time')

    ##--Put column names and content in a dictionary and then convert to a Pandas df--##
    FIREACE_PCASP_df = pd.DataFrame({col: FIREACE_PCASP_data[col] for 
                                    col in PCASP_bin_num})

    ##--Create new column names by rounding the bin center values to the nearest integer--##
    PCASP_new_col_names = PCASP_bin_center.round().astype(int).tolist()

    ##--Rename the PCASP_bins df columns to bin average values--##
    FIREACE_PCASP_data.columns = PCASP_new_col_names

    ######################
    ##--Calc N(2.5-10)--##
    ######################

    ##--Convert to STP!--##
    P_STP = 101325  # Pa
    T_STP = 273.15  # K

    ##--Create empty list for CPC3 particles--##
    FIREACE_CPC3_conc_STP_nonav = []

    ##--Use the NON-AVERAGED data for first calculation--##
    for CPC3, T, P in zip(FIREACE_CPC3_nonav, FIREACE_temperature_nonav, 
    FIREACE_pressure_nonav):
        if np.isnan(CPC3) or np.isnan(T) or np.isnan(P):
            ##--Append with NaN if any input is NaN--##
            FIREACE_CPC3_conc_STP_nonav.append(np.nan)
        else:
            ##--Perform conversion if all inputs are valid--##
            CPC3_conversion = CPC3 * (P_STP / P) * (T / T_STP)
            FIREACE_CPC3_conc_STP_nonav.append(CPC3_conversion)
        
    ##--Create empty list for CPC10 particles--##
    FIREACE_CPC10_conc_STP_nonav = []

    for CPC10, T, P in zip(FIREACE_CPC10_nonav, FIREACE_temperature_nonav, 
    FIREACE_pressure_nonav):
        if np.isnan(CPC10) or np.isnan(T) or np.isnan(P):
            ##--Append with NaN if any input is NaN--##
            FIREACE_CPC10_conc_STP_nonav.append(np.nan)
        else:
            ##--Perform conversion if all inputs are valid--##
            CPC10_conversion = CPC10 * (P_STP / P) * (T / T_STP)
            FIREACE_CPC10_conc_STP_nonav.append(CPC10_conversion)

    ##--Creates a Pandas dataframe for particle data--##
    FIREACE_particle_df_nonav = pd.DataFrame({'Altitude': FIREACE_altitude_nonav, 
                                       'Latitude': FIREACE_latitude_nonav,
                                       'CPC3_conc':FIREACE_CPC3_conc_STP_nonav, 
                                       'CPC10_conc': FIREACE_CPC10_conc_STP_nonav})

    ##--Calculate N3-10 particles--##
    FIREACE_nuc_particles_nonav = (FIREACE_particle_df_nonav['CPC3_conc'] - 
                            FIREACE_particle_df_nonav['CPC10_conc'])

    ##--Change calculated particle counts less than zero to NaN--##
    FIREACE_nuc_particles_nonav = np.where(FIREACE_nuc_particles_nonav >= 0, 
                                    FIREACE_nuc_particles_nonav, np.nan)

    ##--Add nucleating particles to df--##
    FIREACE_particle_df_nonav['n_3_10'] = FIREACE_nuc_particles_nonav
    
    ##--Repeat calculation for AVERAGED CPC data--##
    ##--Create empty list for CPC3 particles--##
    FIREACE_CPC3_conc_STP = []

    ##--Use the NON-AVERAGED data for first calculation--##
    for CPC3, T, P in zip(FIREACE_CPC3_data, FIREACE_temperature, FIREACE_pressure):
        if np.isnan(CPC3) or np.isnan(T) or np.isnan(P):
            ##--Append with NaN if any input is NaN--##
            FIREACE_CPC3_conc_STP.append(np.nan)
        else:
            ##--Perform conversion if all inputs are valid--##
            CPC3_conversion = CPC3 * (P_STP / P) * (T / T_STP)
            FIREACE_CPC3_conc_STP.append(CPC3_conversion)
        
    ##--Create empty list for CPC10 particles--##
    FIREACE_CPC10_conc_STP = []

    for CPC10, T, P in zip(FIREACE_CPC10_data, FIREACE_temperature, FIREACE_pressure):
        if np.isnan(CPC10) or np.isnan(T) or np.isnan(P):
            ##--Append with NaN if any input is NaN--##
            FIREACE_CPC10_conc_STP.append(np.nan)
        else:
            ##--Perform conversion if all inputs are valid--##
            CPC10_conversion = CPC10 * (P_STP / P) * (T / T_STP)
            FIREACE_CPC10_conc_STP.append(CPC10_conversion)

    ##--Creates a Pandas dataframe for particle data--##
    FIREACE_particle_df = pd.DataFrame({'Altitude': FIREACE_altitude, 
                                       'Latitude': FIREACE_latitude,
                                       'CPC3_conc':FIREACE_CPC3_conc_STP, 
                                       'CPC10_conc': FIREACE_CPC10_conc_STP})

    ##--Calculate N3-10 particles--##
    FIREACE_nuc_particles = (FIREACE_particle_df['CPC3_conc'] - 
                            FIREACE_particle_df['CPC10_conc'])

    ##--Change calculated particle counts less than zero to NaN--##
    FIREACE_nuc_particles = np.where(FIREACE_nuc_particles >= 0, 
                                    FIREACE_nuc_particles, np.nan)
    
    ##--Append nucleating particles to averaged df--##
    FIREACE_particle_df["n_3_10"] = FIREACE_nuc_particles

    ############################
    ##--Normalize PCASP data--##
    ############################

    ##--Calculate dlogDp for each bin in numpy array--##
    dlogDp = np.log(PCASP_upper_bound.values) - np.log(PCASP_lower_bound.values)

    ##--Get only particle count data (excluding 'Time')--##
    FIREACE_PCASP_particle_counts = FIREACE_PCASP_data.loc[:, PCASP_new_col_names]

    ##--Normalize counts by dividing by dlogDp across all rows--##
    FIREACE_PCASP_dNdlogDp = FIREACE_PCASP_data.divide(dlogDp, axis=1)

    ##--Create empty list for PCASP particles--##
    FIREACE_PCASP_STP = []

    for PCASP, T, P in zip(FIREACE_PCASP_dNdlogDp.values, 
    FIREACE_data['Temperature']+273.15, FIREACE_data['Pressure']*100):
        if np.isnan(T) or np.isnan(P):
            ##--Append with NaN if any input is NaN--##
            FIREACE_PCASP_STP.append([np.nan]*len(PCASP))
        else:
            ##--Perform conversion if all inputs are valid--##
            corrected_PCASP = PCASP * (P_STP / P) * (T / T_STP)
            FIREACE_PCASP_STP.append(corrected_PCASP)

    ##--Convert back to DataFrame with same columns and index--##
    FIREACE_PCASP_STP = pd.DataFrame(FIREACE_PCASP_STP, 
        columns=FIREACE_PCASP_dNdlogDp.columns, index=FIREACE_particle_df.index)

    ##--Add PCASP data to the dataframe--##
    FIREACE_particle_df = pd.concat([FIREACE_particle_df, FIREACE_PCASP_STP], axis=1)

    ##--Add PCASP total counts to the dataframe--##
    FIREACE_particle_df['PCTcon'] = FIREACE_data['PCTcon']

    ######################
    ##--Calc N(10-130)--##
    ######################

    ##--Calculate particles below UHSAS lower cutoff--##
    FIREACE_n_10_130 = (FIREACE_particle_df['CPC10_conc'] - 
                       FIREACE_particle_df['PCTcon'])

    ##--Change calculated particle counts less than zero to NaN--##
    FIREACE_n_10_130 = np.where(FIREACE_n_10_130 >= 0, FIREACE_n_10_130, np.nan)

    ##--Put N(10-130) bin center in a df--##
    FIREACE_n_10_130_center = pd.DataFrame([70])

    FIREACE_particle_df['n_10_130'] = FIREACE_n_10_130

    ##--Compute TOTAL counts from all size bins combined--##
    FIREACE_particle_df['Total_particles_STP'] = (FIREACE_particle_df['n_3_10'].fillna(0) + 
          FIREACE_particle_df['n_10_130'].fillna(0) + 
          FIREACE_particle_df['PCTcon'].fillna(0))

    ##--Calculate potential temperature--##
    FIREACE_ptemp = calc_ptemp(FIREACE_temperature, FIREACE_pressure)
    
    ##--Also calculate ptemp using nonaveraged data--##
    FIREACE_ptemp_nonav = calc_ptemp(FIREACE_temperature_nonav, FIREACE_pressure_nonav)

    ###########################
    ##--Wrangle binned data--##
    ###########################

    ##--Append Nucleating and Aitken lists with dataframes--##
    ##--Use the NON-AVERAGED data for nulceation mode--##
    FIREACE_nucleating_df = pd.DataFrame({'nucleating': FIREACE_nuc_particles_nonav,
                              'latitude': FIREACE_latitude_nonav, 
                              'PTemp': FIREACE_ptemp_nonav, 
                              'time': FIREACE_time_nonav}).set_index(FIREACE_time_nonav)
    FIREACE_nucleating_dfs.append(FIREACE_nucleating_df)
    
    ##--Convert aitken mode data to dataframe--##
    FIREACE_aitken_df = pd.DataFrame({'aitken': FIREACE_n_10_130,
                              'latitude': FIREACE_latitude, 
                              'PTemp': FIREACE_ptemp, 
                              'time': FIREACE_time}).set_index(FIREACE_time)
    
    ##--Append list of dataframes with ATom aitken data--##
    FIREACE_aitken_dfs.append(FIREACE_aitken_df)

    ##--Concatenate bin edges--##
    FIREACE_combined_bin_edges = np.concatenate([
        [2.5],      # start of first bin
        [10],       # upper edge of N(2.5-10), also lower of next
        [130],       # upper edge of N(10-130), also lower of next
        PCASP_upper_bound.values,  # PCASP bins continue from 130
    ])

    FIREACE_time_averaged = FIREACE_data['Time']

    ##--Calculate time edges for each bin--##
    FIREACE_time_step = FIREACE_time_averaged.iloc[1] - FIREACE_time_averaged.iloc[0]  
    FIREACE_time_edges = np.append(FIREACE_time_averaged, 
                        FIREACE_time_averaged.iloc[-1] + FIREACE_time_step)  # length N + 1

    ##--Concatenate bin centers and reindex--##
    bin_centers = pd.concat([FIREACE_n_10_130_center, 
                    PCASP_bin_center], axis=0).reset_index(drop=True)

    ##--Place all binned data in a single df--##
    FIREACE_all_bins_aligned = FIREACE_PCASP_STP
    FIREACE_all_bins_aligned['6.25'] = FIREACE_particle_df['n_3_10']
    FIREACE_all_bins_aligned['70'] = FIREACE_particle_df['n_10_130']

    ##--Specify time index--##
    FIREACE_time_index = FIREACE_data['Time']  

    ##--Ensure particle bin dataframes are indexed to time_index and properly named--##
    FIREACE_diameter_df = {}

    for col in FIREACE_all_bins_aligned.columns:
        FIREACE_diameter_df[str(col)] = pd.DataFrame(
            FIREACE_all_bins_aligned[col].values,
            index=FIREACE_all_bins_aligned.index, columns=[str(col)])

    ##--Append diameter dfs--##
    FIREACE_diameter_dfs.append(FIREACE_diameter_df)
    
    ##--Append conditions list--##
    FIREACE_conditions_dfs.append(pd.DataFrame({'temperature': 
        FIREACE_temperature, 'pressure': FIREACE_pressure, 
        'latitude': FIREACE_latitude}))
    
######################################
##--Condensation sink calculations--##
######################################

##--Run this as a function to easily iterate through all four campaigns--##
def condensation_sink(temperature, pressure, latitude, time_index, diameter_dfs):

    ##--Constants--##
    
    R = 8.314 # Ideal gas constant (m^3*Pa*K^-1*mol^-1)
    ##--H2SO4 kinetic diam: lifted from Williamson et al for now (avg of their values)--##
    Ds = 5.49E-10 # in m
    ##--Sticking coefficient - fair to assume unity for H2SO4--##
    alpha = 1
    ##--Boltzmann--##
    k = 1.38E-23 # J/K
    ##--Sutherland's law for dynamic viscosity--##
    C = 1.458E-6 # kg/ms*sqrt(K)
    S = 110.4 # K
    
    ##--Variables--##
    
    ##--Convert temperature and pressure from numpy array to dataframe to subvert errors--##
    temperature_series = pd.Series(temperature, index=time_index)
    pressure_series = pd.Series(pressure, index=time_index)
    latitude_series = pd.Series(latitude, index=time_index)
    
    ##--Loop through dfs in diameter_dfs and calculate needed variables for each bin--##
    ##--Store in series initialized at zero--##
    condensation_sink = pd.Series(0, index=time_index)
    
    for diameter, df in diameter_dfs.items():
        
        df = df.reindex(time_index)
        
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
    
    ##--Calculate potential temperature--##
    ##--Constants--##
    p_0 = 1E5 # Reference pressure in Pa (1000 hPa)
    k = 0.286 # Poisson constant for dry air

    potential_temp = temperature_series * (p_0 / pressure_series) ** k
        
    ##--Constrain ATom ptemp to range of other three campaigns--##
    potential_temp[potential_temp > 310] = np.nan

    ##--Place results in a dataframe--##
    condensation_sink = pd.DataFrame({
        'Condensation_Sink': condensation_sink,
        'PTemp': potential_temp,
        'latitude': latitude_series})
    
    ##--Return the dataframe after running the function--##
    return condensation_sink


##--RUN FUNCTION TO COMPUTE SINKS--##

##--For ATom--##
ATom_condensation_sinks = []

for diameter_df, conditions in zip(ATom_diameter_dfs, ATom_conditions_dfs):

    ##--Pull variables from conditions dfs--##
    temperature = conditions['temperature']
    pressure = conditions['pressure']
    latitude = conditions['latitude']
    time_index = conditions.index

    ##--Input into the function--##
    ATom_result = condensation_sink(temperature, pressure, latitude, time_index,
        diameter_df)

    ##--Append the result to the established list--##
    ATom_condensation_sinks.append(ATom_result)
    
##--Repeat for NETCARE--##
NETCARE_condensation_sinks = []

for diameter_df, conditions in zip(NETCARE_diameter_dfs, NETCARE_conditions_dfs):
    
    temperature = conditions['temperature']
    pressure = conditions['pressure']
    latitude = conditions['latitude']
    time_index = conditions.index
    
    result = condensation_sink(temperature, pressure, latitude, time_index, 
        diameter_df)
    
    NETCARE_condensation_sinks.append(result)

##--Repeat for FIREACE--##
FIREACE_condensation_sinks = []

for diameter_df, conditions in zip(FIREACE_diameter_dfs, FIREACE_conditions_dfs):
    
    temperature = conditions['temperature']
    pressure = conditions['pressure']
    latitude = conditions['latitude']
    time_index = conditions.index
    
    result = condensation_sink(temperature, pressure, latitude, time_index, 
        diameter_df)
    
    FIREACE_condensation_sinks.append(result)

############################
##--Create 2D histograms--##
############################

##--Create GLOBAL bin edges for consistency--##
global_lat_edges = np.linspace(66.5, 86.5, num_bins_lat + 1)
global_ptemp_edges = np.linspace(235, 310, num_bins_ptemp + 1)

##--Define a function for calculating 2d MEDIAN values--##
##--Assistance from GPT-5 model for constructing functions--##
def compute_2d_median(df_list, value_col, lat_edges, ptemp_edges):

    ##--Place dataframes in a list if they exist--##
    if isinstance(df_list, pd.DataFrame):
        df_list = [df_list]

    ##--Concatenate values from all input dataframes--##
    all_lat = np.concatenate([df['latitude'].values for df in df_list])
    all_ptemp = np.concatenate([df['PTemp'].values for df in df_list])
    all_val = np.concatenate([df[value_col].values for df in df_list])

    ##--Create a mask of the nan values--##
    mask = (~np.isnan(all_lat) & ~np.isnan(all_ptemp) & ~np.isnan(all_val))
    
    ##--Handle empty plots with no data--##
    if mask.sum() == 0:
     ##--Create the plot using the edges and fill with NaNs--##
     return np.full((len(lat_edges)-1, len(ptemp_edges)-1), np.nan)

    ##--Pull the median stat per bin, ignoring all other outputs--##
    stat, _, _, _ = binned_statistic_2d(all_lat[mask], all_ptemp[mask],
        all_val[mask], statistic="median", bins=[lat_edges, ptemp_edges])

    ##--Function returns binned median statistic--##
    return stat

##--Define a separate but similar function for calcualting 2d SUM values--##
##--Very similar to function above, but returns total count of sig values--##
def compute_2d_sigfraction(df_list, value_col, lat_edges, ptemp_edges, sig_thresh):

    ##--Check that the df list exists and ensure it's a list--##
    if isinstance(df_list, pd.DataFrame):
        df_list = [df_list]

    ##--Concatenate data as before--##
    all_lat = np.concatenate([df['latitude'].values for df in df_list])
    all_ptemp = np.concatenate([df['PTemp'].values for df in df_list])
    all_val = np.concatenate([df[value_col].values for df in df_list])

    ##--Identify the mask--##
    mask = (~np.isnan(all_lat) & ~np.isnan(all_ptemp) & ~np.isnan(all_val))
    
    ##--Apply the mask--##
    lat = all_lat[mask]
    ptemp = all_ptemp[mask]
    val = all_val[mask]

    ##--Handle empty plots with no data--##
    if mask.sum() == 0:
     return np.full((len(lat_edges)-1, len(ptemp_edges)-1), np.nan)

    ##--Boolean: 1 if significant nucleation event--##
    ##--Sig event defined as greater than NETCARE 75th p. uncertainty--##
    sig = (val > sig_thresh).astype(int)
    
    ##--Count the number of significant events per bin--##
    sig_count, _, _, _ = binned_statistic_2d(lat, ptemp, sig, statistic="sum",
        bins=[lat_edges, ptemp_edges])

    ##--Count the total number of data points in each bin, sig or not--##
    total_count, _, _, _ = binned_statistic_2d(lat, ptemp, None, statistic="count",
        bins=[lat_edges, ptemp_edges])

    ##--Calculate the fraction of significant events, ignoring invalid div--##
    with np.errstate(divide='ignore', invalid='ignore'):
        frac = sig_count / total_count

    ##--Discard times when the total count is zero--##
    frac[total_count == 0] = np.nan
    
    ##--Also discard times when fractions are drawn from too few data points--##
    min_points = 3
    frac[total_count < min_points] = np.nan

    ##--Return the filtered significance fraction--##
    return frac

##--Specify which campaigns to run the functions on, PAMARCMiP has no nuc--##
campaigns = [
    ("2018", ATom_nucleating_dfs, ATom_aitken_dfs, ATom_condensation_sinks),
    ("2015", NETCARE_nucleating_dfs, NETCARE_aitken_dfs, NETCARE_condensation_sinks),
    ("1998", FIREACE_nucleating_dfs, FIREACE_aitken_dfs, FIREACE_condensation_sinks)]

################
##--Plotting--##
################

##--Select light->dark colormaps for nuc, ait, and cs--##
nuc_cmap = cm.cm.bamako_r
ait_cmap = cm.cm.lapaz_r
cs_cmap = cm.cm.batlow_r

##--Create a 3x3 figure--## 
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15), sharex=True,
    sharey=True, constrained_layout=True)

##--Iterate through each row of the three dfs per campaign--##
for row, (name, nuc, ait, cs) in enumerate(campaigns):
    
    ##--Run function to get binned median nucleation data--##
    nuc_stat = compute_2d_median(nuc, "nucleating", global_lat_edges,
        global_ptemp_edges)
    
    ##--Set a threshold for significant nucleation datapoint--##
    ##--This is the 75th percentile median NETCARE uncertainty--##
    nuc_sig_thresh = 134

    ##--Compute the fraction that is significant--##
    nuc_sigfraction = compute_2d_sigfraction(
        nuc, "nucleating",
        global_lat_edges,
        global_ptemp_edges,
        nuc_sig_thresh)
    
    ##--Set a threshold value for median nuc hotspot grid cell--##
    nuc_totalsigcount_thresh = 0.25 # count >25% frac as significant

    ##--Mask using the threshold fraction--##
    sig_mask = nuc_sigfraction > nuc_totalsigcount_thresh
    
    ##--Pull the cells that pass the mask--##
    ##--Need to save theses for reuse on Fig 3: Conditions Master Curtain--##
    highlight_cells = np.where(sig_mask)
    
    ##--Want to save the highlighted cells in the 'data' folder--##
    data_dir = root / "data" 
    
    ##--Need to a make sure the data folder exists first--##
    data_dir.mkdir(exist_ok=True)
    
    ##--SAVE the cells of interest for use in conditions figure--##
    ##--Save for each campaign (use f-string to save the name)--##
    np.save(data_dir/f"highlight_cells_{name}.npy", highlight_cells)
    
    ##--Compute median per bin for Aitken mode--##        
    ait_stat = compute_2d_median(ait, "aitken",
                                 global_lat_edges, global_ptemp_edges)

    ##--Compute median per bin for condensation sink--##
    cs_stat = compute_2d_median(cs, "Condensation_Sink",
                                global_lat_edges, global_ptemp_edges)

    ##--Create plot infrastructure by column--##
    ##--Need to transpose--##
    ##--Nucleation column--##
    m1 = axes[row,0].pcolormesh(global_lat_edges, global_ptemp_edges,
        nuc_stat.T, shading="auto", cmap=nuc_cmap, vmin=0, vmax=500)    
    
    ##--Aitken column--##
    m2 = axes[row,1].pcolormesh(global_lat_edges, global_ptemp_edges,
        ait_stat.T, shading="auto", cmap=ait_cmap, vmin=0, vmax=1000)
    
    ##--cs column--##
    m3 = axes[row,2].pcolormesh(global_lat_edges, global_ptemp_edges,
        cs_stat.T, shading="auto", cmap=cs_cmap,
        ##--Use a log scale as cs spans orders of magnitude--##
        norm=mcolors.LogNorm(vmin=0.0000001, vmax=0.01))
    
    ##--Add boxes for sig nucleating counts > threshold--##
    ax1 = axes[row, 0]
    ax2 = axes[row, 1]
    ax3 = axes[row, 2]
    
    ##--Draw the actual significant boxes specified earlier--##
    ##--Do for each bin--##
    ##--Assistance from GPT-5 model for drawing bin edges--##
    for i in range(nuc_stat.shape[0]):      # ptemp bins
        for j in range(nuc_stat.shape[1]):  # lat bins
            
            ##--Select bins that pass the mask--##
            if sig_mask[j, i]:  # Swapped order because the plot is transposed  
                
                ##--Grab the x and y bin dimensions--##
                x0 = global_lat_edges[j]
                x1 = global_lat_edges[j+1]
                y0 = global_ptemp_edges[i]
                y1 = global_ptemp_edges[i+1]
                
                ##--Calculate bin width and height--##
                width = x1 - x0
                height = y1 - y0
                
                ##--Want to create a new rectangle for each plot in column--##
                for ax in [ax1, ax2, ax3]:
                    
                    ##--Use patches to draw--##
                    rect = patches.Rectangle((x0, y0), width, height, fill=False,
                        edgecolor='deeppink', linewidth=2.5, zorder=10)
                    
                    ax.add_patch(rect)
                    
    ##--Set ptemp label for each row (axes share ylabel)--##
    ax1.set_ylabel("θ (K)", fontsize=22)
    
##--Add polar dome boundaries to NETCARE plots--##
for ax in axes[1,:]:
    ax.axhline(y=285, color='k', linestyle='--', linewidth=1)
    ax.axhline(y=299, color='k', linestyle='--', linewidth=1)
    
##--Set title for each column--##
axes[0,0].set_title("Nucleation Mode", fontsize=26)
axes[0,1].set_title("Aitken Mode", fontsize=26)
axes[0,2].set_title("Condensation Sink", fontsize=26)

##--Select all x axes and config latitude--##
for ax in axes[-1,:]:
    ax.set_xlabel("Latitude (°)", fontsize=22)
    ax.tick_params(labelsize=18)
    ax.set_xlim(64.5, 87)
    ax.set_xticks([65, 70, 75, 80, 85])

##--Select all y axes and config ptemp--##    
for ax in axes[:,0]:
    ax.tick_params(axis='y', labelsize=18)
    ax.set_ylim(230, 315)
    
##--Add sup labels for campaign year--##
axes[0, 0].text(-0.35, 0.5, "2018", verticalalignment='center', rotation=90, 
                fontsize=26, weight='bold', transform=axes[0,0].transAxes)
axes[1, 0].text(-0.35, 0.5, "2015", verticalalignment='center', fontsize=26, 
                rotation=90, weight='bold', transform=axes[1,0].transAxes)
axes[2, 0].text(-0.35, 0.5, "1998", verticalalignment='center', fontsize=26, 
                rotation=90, weight='bold', transform=axes[2,0].transAxes)
    
##--Add labels with particle bin sizes for each group--##
##--ATom--##
axes[0, 0].text(0.05, 0.05, "2.7-12 nm", fontsize=18, transform=axes[0,0].transAxes)
axes[0, 1].text(0.05, 0.05, "12-60 nm", fontsize=18, transform=axes[0,1].transAxes)
axes[0, 2].text(0.05, 0.05, "12+ nm", fontsize=18, transform=axes[0,2].transAxes)

##--NETCARE--##
axes[1, 0].text(0.05, 0.05, "2.5-10 nm", fontsize=18, transform=axes[1,0].transAxes)
axes[1, 1].text(0.05, 0.05, "10-85 nm", fontsize=18, transform=axes[1,1].transAxes)
axes[1, 2].text(0.05, 0.05, "10+ nm", fontsize=18, transform=axes[1,2].transAxes)

##--FIRE-ACE--##
axes[2, 0].text(0.05, 0.05, "2.5-10 nm", fontsize=18, transform=axes[2,0].transAxes)
axes[2, 1].text(0.05, 0.05, "10-130 nm", fontsize=18, transform=axes[2,1].transAxes)
axes[2, 2].text(0.05, 0.05, "10+ nm", fontsize=18, transform=axes[2,2].transAxes)

##--Create colorbar object for nucleating data--## 
##--Specify the exact ticks here so I can rename them as strings later--##  
nuc_cbar = fig.colorbar(m1, ax=axes[:,0], ticks=[0, 100, 200, 300, 400, 500], 
                        location='bottom', pad=0.01, shrink=0.95)

##--Apply the colorbar--##
nuc_cbar.set_label(label="Nucleation Mode Particles (#/cm$^{3}$)", fontsize=18, labelpad=15)

##--Add line with NETCARE 134 counts/cm^3 significance cutoff to nucleation mode colorbar--##
nuc_cbar.ax.axvline(x=134, color='black', linewidth=3)

##--Add + to last tick label: re-label ticks with strings--##
##--Setting this limit makes better use of colorbar range--##
nuc_cbar.set_ticklabels(["0", "100", "200", "300", "400", "500+"], 
                        fontsize=18, rotation=60)

##--Do the same for the Aitken mode colorbar--##
ait_cbar = fig.colorbar(m2, ax=axes[:,1], ticks=[0, 200, 400, 600, 800, 1000], 
                        location='bottom', pad=0.01, shrink=0.95)
ait_cbar.set_label(label="Aitken Mode Particles (#/cm$^{3}$)", fontsize=18, labelpad=5)

##--Add + to last tick label by re-labeling as strings--##
ait_cbar.set_ticklabels(["0", "200", "400", "600", "800", "1000+"],
                        fontsize=18, rotation=60)

##--Create colorbar object for the condensation sink--##
cs_cbar = fig.colorbar(m3, ax=axes[:,2], ticks=[1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2], 
                       location='bottom', pad=0.01, shrink=0.95)

##--No need to re-label this colorbar with strings since whole range is used--##
cs_cbar.set_label(label="Condensation Sink (s$^{-1}$)", fontsize=18, labelpad=21)
cs_cbar.ax.tick_params(labelsize=18, rotation=60)

##--Flatten the axes to an array for iteration--##
axes_flat = axes.flatten()

##--Add lowercase suplot labels--##
for n, ax in enumerate(axes_flat):
    ##--iterate through letters using ascii--##
    ##--use transAxes to place letter using axes coords--##
    ax.text(0.95, 0.98, string.ascii_lowercase[n], transform=ax.transAxes, 
            size=22, weight='bold', va='top', ha='right') 
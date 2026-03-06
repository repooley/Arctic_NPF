# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 11:19:45 2026

@author: repooley
"""


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

###################
##--User inputs--##
###################

##--Set the base directories to project folder--##
ATom_directory = r"C:\Users\repooley\REP_PhD\Arctic_NPF\ATom2018\data\raw"
NETCARE_directory = r"C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data\raw"
PAMARCMiP_directory = r"C:\Users\repooley\REP_PhD\Arctic_NPF\PAMARCMiP2012\data\raw"
FIREACE_directory = r"C:\Users\repooley\REP_PhD\Arctic_NPF\FIREACE1998\data\raw"

##--Choose which flights to analyze here!--##
##--ATom--##
ATom_to_analyze = ["Flight2", "Flight10", "Flight11", "Flight12"]

##--NETCARE--##
NETCARE_to_analyze = ["Flight2", "Flight3", "Flight4", "Flight5", "Flight6", 
                      'Flight7', 'Flight8', 'Flight9', 'Flight10']

##--PAMARCMiP--##
PAMARCMiP_to_analyze = ["Flight1", "Flight2", "Flight3", "Flight6", 
                      "Flight7", "Flight8", "Flight9"]

##--FIRE-ACE--##
FIREACE_to_analyze = ["Flight3",  "Flight7", "Flight8", "Flight9", "Flight10", 
                      "Flight11", "Flight12", "Flight13", "Flight14", 
                      "Flight15", "Flight16", "Flight17", "Flight18"]

##--Set number of bins for latitude and potential temperature--##
num_bins_lat = 12
num_bins_ptemp = 12

##--Base output path for figures in directory--##
#output_path = r"C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data\processed\CurtainPlots\CondensationSink"

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
    dataset = icartt.Dataset(find_files(ATom_directory, flight, "MER")[0])

    #################
    ##--Pull data--##
    #################

    altitude = dataset.data['G_ALT'] # in m (not sure if this is best one)
    latitude = dataset.data['LAT_AMSSD'] # deg
    temperature = dataset.data['T'] # in K
    pressure = dataset.data['P'] * 100 # in Pa
    RH = dataset.data['Relative_Humidity'] # wrt water, percent
    time =dataset.data['UTC_Start'] # seconds since midnight UTC
    nucleating = dataset.data['N_nucl_AMP'] # num/cm^3 STP (2.7-12 nm)
    aitken = dataset.data['N_aitken_AMP'] # num/cm^3 STP (12-60 nm)
    accumulation = dataset.data['N_accum_AMP'] # num/cm^3 STP (60 nm - 0.5 um)
    coarse = dataset.data['N_coarse_AMP'] # num/cm^3 STP (0.5 um - 4.8 um)
    
    ##--Constrain latitude to the Arctic region--##
    latitude[latitude < 66.5] = np.nan
    
    ##--There are notable outliers in the nucleating data--##
    
    ##--First convert to a series for calc--##
    nucleating_series = pd.Series(nucleating)
    
    ##--REMOVE OUTLIERS above 99th percentile--##
    p = 0.99
    
    ##--Compute threshold for each UHSAS column--##
    nucleating_thresh = nucleating_series.quantile(p)
    
    ##--keep only rows where each bin is below its threshold--##
    nucleating_filtered = nucleating_series[nucleating_series.le(nucleating_thresh)]
    
    nucleating_filtered = nucleating_series.mask(
    nucleating_series <= nucleating_thresh)
    
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
        
    ##--Convert ptemp to np array--##
    potential_temp = np.array(potential_temp)
        
    ##--Constrain ptemp to range of other three campaigns--##
    potential_temp[potential_temp > 310] = np.nan
        
    ##--Append Nucleating and Aitken lists with dataframes--##
    ##--Convert nucleating data to dataframe--##
    nucleating_df = pd.DataFrame({'nucleating': nucleating_series, 
                                  'latitude': latitude, 
                                  'PTemp': potential_temp, 
                                  'time': time}).set_index(time)
    
    ##--Append list of dataframes with ATom nucleating data--##
    ATom_nucleating_dfs.append(nucleating_df)
    
    ##--Convert aitken mode data to dataframe--##
    aitken_df = pd.DataFrame({'aitken': aitken,
                              'latitude': latitude, 
                              'PTemp': potential_temp, 
                              'time': time}).set_index(time)
    
    ##--Append list of dataframes with ATom aitken data--##
    ATom_aitken_dfs.append(aitken_df)
        
    ###########################
    ##--Wrangle binned data--##
    ###########################
    
    ##--Concatenate bin edges--##
    combined_bin_edges = np.concatenate([
        [12],       # upper edge of N(2.7-12), also lower of next
        [60],       # upper edge of N(12-60), also lower of next
        [500],      # upper edge of N(60-500), also lower of next
        [4800]      # upper edge of final bin
        ])
    
    ##--Concatenate bin centers--##
    bin_centers = np.concatenate([
        [36], 
        [280],
        [2650]
        ])
    
    ##--Place all binned data in a single df--##
    all_bins_aligned = pd.concat([pd.DataFrame({'36':aitken, '280':accumulation, 
                                        '2650':coarse}, index=time)], axis=1)
    
    total_particle_count = all_bins_aligned.sum(axis=1, numeric_only=True) 
    
    ##--Create a dictionary to store each column as a separate dataframe, col names are keys--##
    diameter_dfs = {col: pd.DataFrame({col: all_bins_aligned[col]}) for col in all_bins_aligned.columns}
    
    ATom_diameter_dfs.append(diameter_dfs)
    
    ATom_conditions_dfs.append(pd.DataFrame({'temperature': temperature, 
                    'pressure': pressure, 'latitude': latitude}, index=time))
    
##--NETCARE--##
NETCARE_nucleating_dfs = []
NETCARE_aitken_dfs = []
NETCARE_diameter_dfs = []
NETCARE_conditions_dfs = []
 
##--Loop through each flight, pulling and analyzing data--##
for flight in NETCARE_to_analyze:
    ##--Follow which flight is processing--##
    print(f"Processing {flight}...")

    ##--Pull meteorological data from AIMMS monitoring system--##
    aimms_files = find_files(NETCARE_directory, flight, "AIMMS_POLAR6")
    if aimms_files:
        aimms = icartt.Dataset(aimms_files[0])
    else:
        print(f"No AIMMS_POLAR6 file found for {flight}. Skipping...")
        continue  
 
    ##--Pull CPC files--##
    CPC10_files = find_files(NETCARE_directory, flight, 'CPC3772')
    CPC3_files = find_files(NETCARE_directory, flight, 'CPC3776')
 
    if CPC10_files and CPC3_files:
        ##--Make variables containing all CPC dataset objects--##
        CPC10 = icartt.Dataset(CPC10_files[0])
        CPC3 = icartt.Dataset(CPC3_files[0])
    else:
        print(f"Missing CPC data for {flight}. Skipping...")
        continue
    
    ##--Pull UHSAS files--##
    UHSAS_files = find_files(NETCARE_directory, flight, "UHSAS")
    if UHSAS_files:
        UHSAS = icartt.Dataset(UHSAS_files[0])
    else:
        print(f"No UHSAS_POLAR6 file found for {flight}. Skipping...")
        continue  
    
    ##--Pull OPC files--##
    OPC_files = find_files(NETCARE_directory, flight, 'OPC')
 
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
    
    ##--Constrain latitude to the Arctic region--##
    latitude[latitude < 66.5] = np.nan
    
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
    UHSAS_bins = pd.read_csv(r"C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data\raw\NETCARE2015_UHSAS_bins.csv")

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

    ##--OPC samples every six seconds. Most rows are NaN--##
    ##--Forward-fill NaN values to propagate last valid reading--##
    ##--Limit forward filling to 5 NaN rows--##
    OPC_bins_filled = OPC_bins_aligned.ffill(limit=5)

    ###############################
    ##--De-Normalize UHSAS Data--##
    ###############################
    
    ##--For total count calculation--##
    
    ##--Calculate dlogDp for UHSAS bins--##
    UHSAS_dlogDp = np.log(UHSAS_upper_bound.values) - np.log(UHSAS_lower_bound.values)
    
    ##--Get only particle count data (excluding 'Time')--##
    UHSAS_particle_counts = UHSAS_bins_aligned.loc[:, UHSAS_new_col_names]  # Adjust column names as needed
    
    ##--De-Normalize counts by multiplying by dlogDp across all rows--##
    UHSAS_denorm_counts = UHSAS_particle_counts.multiply(UHSAS_dlogDp, axis=1)
    
    ##--Take out of STP--##
    P_STP = 101325  # Pa
    T_STP = 273.15  # K
    
    ##--Reset index of UHSAS_abs_counts to align with time--##
    min_length = min(len(UHSAS_time), len(UHSAS_denorm_counts))
    UHSAS_time = UHSAS_time[:min_length]
    UHSAS_denorm_counts = UHSAS_denorm_counts.iloc[:min_length]
    
    ##########################
    ##--Normalize OPC Data--##
    ##########################
    
    ##--Use the de-normalized values for calculating NPF--##

    ##--Calculate dlogDp for each bin in numpy array--##
    dlogDp = np.log(OPC_upper_bound.values) - np.log(OPC_lower_bound.values)
    
    ##--Get only particle count data (excluding 'Time')--##
    OPC_particle_counts = OPC_bins_filled.loc[:, OPC_new_col_names]
    
    ##--Normalize counts by dividing by dlogDp across all rows--##
    OPC_dNdlogDp = OPC_bins_filled.divide(dlogDp, axis=1)
    
    ##--Convert to STP!--##
    P_STP = 101325  # Pa
    T_STP = 273.15  # K
    
    ##--Create empty list for OPC particles--##
    OPC_conc_STP_norm = []
    
    for OPC, T, P in zip(OPC_dNdlogDp.values, temperature, pressure):
        if np.isnan(T) or np.isnan(P):
            ##--Append with NaN if any input is NaN--##
            OPC_conc_STP_norm.append([np.nan]*len(OPC))
        else:
            ##--Perform conversion if all inputs are valid--##
            corrected_OPC = OPC * (P_STP / P) * (T / T_STP)
            OPC_conc_STP_norm.append(corrected_OPC)
    
    ##--Convert back to DataFrame with same columns and index--##
    OPC_conc_STP_norm = pd.DataFrame(OPC_conc_STP_norm, columns=OPC_dNdlogDp.columns, index=OPC_dNdlogDp.index)
    
    ##--Repeat for DENORM OPC--##
    OPC_conc_STP_denorm = []
    
    for OPC, T, P in zip(OPC_particle_counts.values, temperature, pressure):
        if np.isnan(T) or np.isnan(P):
            ##--Append with NaN if any input is NaN--##
            OPC_conc_STP_denorm.append([np.nan]*len(OPC))
        else:
            ##--Perform conversion if all inputs are valid--##
            corrected_OPC = OPC * (P_STP / P) * (T / T_STP)
            OPC_conc_STP_denorm.append(corrected_OPC)
            
    ##--Convert back to DataFrame with same columns and index--##
    OPC_conc_STP_denorm = pd.DataFrame(OPC_conc_STP_denorm, columns=OPC_particle_counts.columns, index=OPC_particle_counts.index)
    
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
    
    ##--Creates a Pandas dataframe for CPC data--##
    CPC_df = pd.DataFrame({'CPC3_conc':CPC3_conc_STP, 'CPC10_conc': CPC10_conc_STP})
    
    ##--Calculate N3-10 particles--##
    n_3_10 = (CPC_df['CPC3_conc'] - CPC_df['CPC10_conc'])
    
    ##--Change calculated particle counts less than zero to NaN--##
    n_3_10 = np.where(n_3_10 >= 0, n_3_10, np.nan)

    #####################
    ##--Calc N(10-89)--##
    #####################

    ##--Create df with UHSAS total counts--##
    UHSAS_total = pd.DataFrame({'Time': UHSAS_time, 'Total_count': UHSAS_denorm_counts.sum(axis=1)})

    ##--Reindex UHSAS_total df to AIMMS time--##
    UHSAS_total_aligned = UHSAS_total.set_index('Time').reindex(aimms_time)
    
    ##--Same for OPC--##
    OPC_total = OPC_conc_STP_denorm.sum(axis=1)
    
    OPC_total_aligned = pd.DataFrame({'Time': aimms_time, 'Total_count': OPC_total}).set_index('Time')

    ##--Create df with CPC10 counts and set index to time--##
    CPC10_counts = pd.DataFrame({'Time':aimms_time, 'Counts':CPC10_conc_aligned}).set_index('Time')

    ###--Calculate particles below UHSAS lower cutoff--##
    n_10_89 = (CPC10_counts['Counts'] - (UHSAS_total_aligned['Total_count'] + OPC_total_aligned['Total_count']))

    ##--Change calculated particle counts less than zero to NaN--##
    n_10_89 = np.where(n_10_89 >= 0, n_10_89, np.nan)

    ##--Put N(10-89) bin center in a df--##
    n_10_89_center = pd.DataFrame([49.5])

    ##--Convert n_10_89 to a df--##
    n_10_89 = pd.DataFrame({'49.5': n_10_89, 'time':aimms_time}).set_index('time')

    ##--Change first column name from string to float--##
    n_10_89.columns = [49.5]
    
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
        
    ##--Append Nucleating and Aitken lists with dataframes--##
    ##--Convert nucleating data to dataframe--##
    nucleating_df = pd.DataFrame({'nucleating': n_3_10, 
                                  'latitude': latitude, 
                                  'PTemp': potential_temp, 
                                  'time': aimms_time}).set_index('time')
    
    ##--Append list of dataframes with ATom nucleating data--##
    NETCARE_nucleating_dfs.append(nucleating_df)
    
    ##--Convert aitken mode data to dataframe--##
    aitken_df = pd.DataFrame({'aitken': n_10_89[49.5],
                              'latitude': latitude, 
                              'PTemp': potential_temp, 
                              'time': aimms_time}).set_index('time')
    
    ##--Append list of dataframes with ATom aitken data--##
    NETCARE_aitken_dfs.append(aitken_df)

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
    
    ##--Concatenate bin centers and reindex--##
    bin_centers = pd.concat([n_10_89_center, UHSAS_bin_center, OPC_bin_center], axis=0).reset_index(drop=True)
    
    ##--Place all binned data in a single df--##
    all_bins_aligned = pd.concat([n_10_89, UHSAS_bins_aligned, OPC_conc_STP_norm], axis=1)
    total_particle_count = all_bins_aligned.sum(axis=1, numeric_only=True) 
    
    ##--Create a dictionary to store each column as a separate dataframe, col names are keys--##
    NETCARE_dfs = {col: pd.DataFrame({col: all_bins_aligned[col]}) for col in all_bins_aligned.columns}
    
    ##--Create a list of all the diameter dfs--##
    NETCARE_diameter_dfs.append(NETCARE_dfs)
    
    ##--Place conditions in a separate df--##
    NETCARE_conditions_dfs.append(pd.DataFrame({'temperature': temperature, 
                'pressure': pressure, 'latitude': latitude}, index=aimms_time))
    
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
    data = pd.read_csv(find_files(PAMARCMiP_directory, flight, ".csv")[0])
    
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
    ##--Constrain latitude to the Arctic region--##
    latitude = latitude.where(latitude >= 66.5, np.nan)
    
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
              
    ##--Append Nucleating and Aitken lists with dataframes--##
    nucleating_df = pd.DataFrame({'nucleating': np.full(len(time), np.nan),
                              'latitude': latitude, 
                              'PTemp': potential_temp, 
                              'time': time}).set_index(time)
    PAMARCMiP_nucleating_dfs.append(nucleating_df)
    
    ##--Convert aitken mode data to dataframe--##
    aitken_df = pd.DataFrame({'aitken': n_10_60_df['35'],
                              'latitude': latitude, 
                              'PTemp': potential_temp, 
                              'time': time}).set_index(time)
    
    ##--Append list of dataframes with ATom aitken data--##
    PAMARCMiP_aitken_dfs.append(aitken_df)
    
    ###########################
    ##--Wrangle binned data--##
    ###########################
    
    ##--Concatenate bin edges--##
    combined_bin_edges = np.concatenate([
        [10],       # lower edge of N(10-60)
        [60],       # upper edge of N(10-60), also lower of next
        UHSAS_upper_bound.values,  # UHSAS bins continue from 60
    ])
    
    ##--Concatenate bin centers and reindex--##
    bin_centers = pd.concat([n_10_60_center, UHSAS_bin_center], axis=0).reset_index(drop=True)
    
    ##--Place all binned data in a single df--##
    all_bins_aligned = pd.concat([n_10_60_df['35'], UHSAS_bins_filtered], axis=1)
    total_particle_count = all_bins_aligned.sum(axis=1, numeric_only=True) 
    
    ##--Create a dictionary to store each column as a separate dataframe, col names are keys--##
    diameter_dfs = {col: pd.DataFrame({col: all_bins_aligned[col]}) for col in all_bins_aligned.columns}
    
    ##--Append list of diameter dataframes for PAMARCMiP--##
    PAMARCMiP_diameter_dfs.append(diameter_dfs)
    
    ##--Append list of conditions--##
    ##--Place conditions in a separate df--##
    PAMARCMiP_conditions_dfs.append(pd.DataFrame({'temperature': temperature, 
                            'pressure': pressure, 'latitude': latitude}))
    
FIREACE_nucleating_dfs = []
FIREACE_aitken_dfs = []
FIREACE_diameter_dfs = []
FIREACE_conditions_dfs = []

for flight in FIREACE_to_analyze: 
    
    ##--Pull csv file containing all data--##
    files = find_files(FIREACE_directory, flight, "FIREACE")

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
    
    ##--Constrain latitude to the Arctic region--##
    latitude = latitude.where(latitude >= 66.5, np.nan)

    ##--Particle data, 3 and 10 nm cutoffs, respectively--##
    CPC3_data = data['CN3025'] # Uncorrected data has a flow issue - but corrected not populated for many flights
    CPC10_data = data['CN7610']
    
    ##--Path to csv file containing PCASP bin info--##
    PCASP_bins_path = r"C:\Users\repooley\REP_PhD\Arctic_NPF\FIREACE1998\data\raw\FIREACE1998_PCASP_bins.csv"

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

    ##--Append Nucleating and Aitken lists with dataframes--##
    nucleating_df = pd.DataFrame({'nucleating': nuc_particles,
                              'latitude': latitude, 
                              'PTemp': potential_temp, 
                              'time': time}).set_index(time)
    FIREACE_nucleating_dfs.append(nucleating_df)
    
    ##--Convert aitken mode data to dataframe--##
    aitken_df = pd.DataFrame({'aitken': n_10_130,
                              'latitude': latitude, 
                              'PTemp': potential_temp, 
                              'time': time}).set_index(time)
    
    ##--Append list of dataframes with ATom aitken data--##
    FIREACE_aitken_dfs.append(aitken_df)

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
        diameter_dfs[str(col)] = pd.DataFrame(
            all_bins_aligned[col].values,
            index=all_bins_aligned.index,
            columns=[str(col)]
        )

    ##--Append diameter dfs--##
    FIREACE_diameter_dfs.append(diameter_dfs)
    
    ##--Append conditions list--##
    FIREACE_conditions_dfs.append(pd.DataFrame({'temperature': temperature, 
                                'pressure': pressure, 'latitude': latitude}))
    
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
    
    #######################################
    ##--Calculate potential temperature--##
    #######################################
    
    ##--Constants--##
    p_0 = 1E5 # Reference pressure in Pa (1000 hPa)
    k = 0.286 # Poisson constant for dry air

    potential_temp = temperature_series * (p_0 / pressure_series) ** k
        
    ##--Constrain ATom ptemp to range of other three campaigns--##
    potential_temp[potential_temp > 310] = np.nan

    condensation_sink = pd.DataFrame({
        'Condensation_Sink': condensation_sink,
        'PTemp': potential_temp,
        'latitude': latitude_series})
    
    return condensation_sink


##--RUN FUNCTION TO COMPUTE SINKS--##

##--For ATom--##
ATom_condensation_sinks = []

for diameter_df, conditions in zip(ATom_diameter_dfs, ATom_conditions_dfs):

    temperature = conditions['temperature']
    pressure = conditions['pressure']
    latitude = conditions['latitude']
    time_index = conditions.index

    result = condensation_sink(
        temperature,
        pressure,
        latitude, 
        time_index,
        diameter_df
    )

    ATom_condensation_sinks.append(result)
    
##--For NETCARE--##
NETCARE_condensation_sinks = []

for diameter_df, conditions in zip(NETCARE_diameter_dfs, NETCARE_conditions_dfs):
    
    temperature = conditions['temperature']
    pressure = conditions['pressure']
    latitude = conditions['latitude']
    time_index = conditions.index
    
    result = condensation_sink(
        temperature,
        pressure, 
        latitude,
        time_index, 
        diameter_df)
    
    NETCARE_condensation_sinks.append(result)

##--For PAMARCMiP--##
PAMARCMiP_condensation_sinks = []

for diameter_df, conditions in zip(PAMARCMiP_diameter_dfs, PAMARCMiP_conditions_dfs):
    
    temperature = conditions['temperature']
    pressure = conditions['pressure']
    latitude = conditions['latitude']
    time_index = conditions.index
    
    result = condensation_sink(
        temperature,
        pressure, 
        latitude,
        time_index, 
        diameter_df)
    
    PAMARCMiP_condensation_sinks.append(result)

##--For FIREACE--##
FIREACE_condensation_sinks = []

for diameter_df, conditions in zip(FIREACE_diameter_dfs, FIREACE_conditions_dfs):
    
    temperature = conditions['temperature']
    pressure = conditions['pressure']
    latitude = conditions['latitude']
    time_index = conditions.index
    
    result = condensation_sink(
        temperature,
        pressure, 
        latitude,
        time_index, 
        diameter_df)
    
    FIREACE_condensation_sinks.append(result)


############################
##--Create 2D histograms--##
############################

all_latitudes = np.concatenate([
    np.concatenate([df['latitude'].values for df in ATom_condensation_sinks]),
    np.concatenate([df['latitude'].values for df in NETCARE_condensation_sinks]),
    #np.concatenate([df['latitude'].values for df in PAMARCMiP_condensation_sinks]),
    np.concatenate([df['latitude'].values for df in FIREACE_condensation_sinks])
])

global_lat_edges = np.linspace(
    66.5,
    86.5,
    num_bins_lat + 1
)

# --- Global potential temperature ---
all_ptemps = np.concatenate([
    np.concatenate([df['PTemp'].values for df in ATom_condensation_sinks]),
    np.concatenate([df['PTemp'].values for df in NETCARE_condensation_sinks]),
    #np.concatenate([df['PTemp'].values for df in PAMARCMiP_condensation_sinks]),
    np.concatenate([df['PTemp'].values for df in FIREACE_condensation_sinks])
])

global_ptemp_edges = np.linspace(
    235,
    310,
    num_bins_ptemp + 1
)

def compute_2d_median(df_list, value_col, lat_edges, ptemp_edges):

    if isinstance(df_list, pd.DataFrame):
        df_list = [df_list]

    all_lat = np.concatenate([df['latitude'].values for df in df_list])
    all_ptemp = np.concatenate([df['PTemp'].values for df in df_list])
    all_val = np.concatenate([df[value_col].values for df in df_list])

    mask = (~np.isnan(all_lat) &
            ~np.isnan(all_ptemp) &
            ~np.isnan(all_val))
    
    ##--Handle empty plots with no data--##
    if mask.sum() == 0:
     return np.full(
         (len(lat_edges)-1, len(ptemp_edges)-1),
         np.nan
     )

    stat, _, _, _ = binned_statistic_2d(
        all_lat[mask],
        all_ptemp[mask],
        all_val[mask],
        statistic="median",
        bins=[lat_edges, ptemp_edges]
    )

    return stat

campaigns = [
    ("2018", ATom_nucleating_dfs, ATom_aitken_dfs, ATom_condensation_sinks),
    ("2015", NETCARE_nucleating_dfs, NETCARE_aitken_dfs, NETCARE_condensation_sinks),
    #("2012", PAMARCMiP_nucleating_dfs, PAMARCMiP_aitken_dfs, PAMARCMiP_condensation_sinks),
    ("1998", FIREACE_nucleating_dfs, FIREACE_aitken_dfs, FIREACE_condensation_sinks),
]

################
##--Plotting--##
################

nuc_cmap = cm.cm.bamako_r
ait_cmap = cm.cm.lapaz_r
cs_cmap = cm.cm.batlow_r
 
fig, axes = plt.subplots(
    nrows=3,
    ncols=3,
    figsize=(15, 15),
    sharex=True,
    sharey=True,
    constrained_layout=True)

##--Set a threshold value for median nuc hotspot grid cell--##
nuc_thresh = 100

for row, (name, nuc, ait, cs) in enumerate(campaigns):

    nuc_stat = compute_2d_median(nuc, "nucleating",
                                 global_lat_edges, global_ptemp_edges)
    
    ##--Create a boolean array of values below or above threshold--##
    mask = nuc_stat > nuc_thresh   
            
    ait_stat = compute_2d_median(ait, "aitken",
                                 global_lat_edges, global_ptemp_edges)

    cs_stat = compute_2d_median(cs, "Condensation_Sink",
                                global_lat_edges, global_ptemp_edges)

    m1 = axes[row,0].pcolormesh(
        global_lat_edges,
        global_ptemp_edges,
        nuc_stat.T,
        shading="auto",
        cmap=nuc_cmap,
        vmin=0,
        vmax=500
    )    
    
    m2 = axes[row,1].pcolormesh(
        global_lat_edges,
        global_ptemp_edges,
        ait_stat.T,
        shading="auto",
        cmap=ait_cmap,
        vmin=0, 
        vmax=1000
    )
    
    m3 = axes[row,2].pcolormesh(
        global_lat_edges,
        global_ptemp_edges,
        cs_stat.T,
        shading="auto",
        cmap=cs_cmap,
        norm=mcolors.LogNorm(vmin=0.0000001, vmax=0.01)
    )
    
    ## -- Add boxes for median nucleating > threshold -- ##
    ax1 = axes[row, 0]
    ax2 = axes[row, 1]
    ax3 = axes[row, 2]
    
    for i in range(nuc_stat.shape[0]):      # ptemp bins
        for j in range(nuc_stat.shape[1]):  # lat bins
            
            if mask[j, i]:  # Swapped order because the plot is transposed  
                
                x0 = global_lat_edges[j]
                x1 = global_lat_edges[j+1]
                y0 = global_ptemp_edges[i]
                y1 = global_ptemp_edges[i+1]
                
                width = x1 - x0
                height = y1 - y0
                
                # Create a NEW rectangle for EACH axis
                for ax in [ax1, ax2, ax3]:
                    rect = patches.Rectangle(
                        (x0, y0),
                        width,
                        height,
                        fill=False,
                        edgecolor='deeppink',
                        linewidth=2.5,
                        zorder=10
                    )
                    ax.add_patch(rect)

    ax1.set_ylabel("θ (K)", fontsize=22)
    
##--Add polar dome boundaries to NETCARE plots--##
for ax in axes[1,:]:
    ax.axhline(y=285, color='k', linestyle='--', linewidth=1)
    ax.axhline(y=299, color='k', linestyle='--', linewidth=1)
    

axes[0,0].set_title("Nucleation Mode", fontsize=26)
axes[0,1].set_title("Aitken Mode", fontsize=26)
axes[0,2].set_title("Condensation Sink", fontsize=26)

for ax in axes[-1,:]:
    ax.set_xlabel("Latitude (°)", fontsize=22)
    ax.tick_params(labelsize=18)
    ax.set_xlim(64.5, 87)
    ax.set_xticks([65, 70, 75, 80, 85])
    
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
   
nuc_cbar = fig.colorbar(m1, ax=axes[:,0], ticks=[0, 100, 200, 300, 400, 500], 
                        location='bottom', pad=0.01, shrink=0.95)

nuc_cbar.set_label(label="Nucleation Mode Particles (#/cm$^{3}$)", fontsize=18, labelpad=15)

##--Add + to last tick label--##
nuc_cbar.set_ticklabels(["0", "100", "200", "300", "400", "500+"], 
                        fontsize=18, rotation=60)

ait_cbar = fig.colorbar(m2, ax=axes[:,1], ticks=[0, 200, 400, 600, 800, 1000], 
                        location='bottom', pad=0.01, shrink=0.95)
ait_cbar.set_label(label="Aitken Mode Particles (#/cm$^{3}$)", fontsize=18, labelpad=5)

##--Add + to last tick label--##
ait_cbar.set_ticklabels(["0", "200", "400", "600", "800", "1000+"],
                        fontsize=18, rotation=60)

cs_cbar = fig.colorbar(m3, ax=axes[:,2], ticks=[1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2], 
                       location='bottom', pad=0.01, shrink=0.95)

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

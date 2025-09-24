# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 09:06:44 2025

@author: repooley
"""

import icartt
import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime
import pyarrow

###################
##--User inputs--##
###################
 
##--Set the base directory to project folder--##
directory = r"C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data\raw"

##--Choose which flights to analyze here!--##
flights_to_analyze = ["Flight2", "Flight3", "Flight4", "Flight5", "Flight6", "Flight7", "Flight8", "Flight9", "Flight10"]

#######################################
##--Open ICARTT Files and Pull Data--##
#######################################

##--Pull datasets with zeros not filtered out--##
CPC3_R1 = icartt.Dataset(r"C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data\raw\CPC_R1\CPC3776_Polar6_20150408_R1_L2.ict")    
CPC10_R1 = icartt.Dataset(r'C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data\raw\CPC_R1\CPC3772_Polar6_20150408_R1_L2.ict')
 
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

##--Define a function to pull the start date from the AIMMS files--##
def get_icartt_dates(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        date_line = lines[6]  # Line 7 (0-based index)
    parts = [int(p.strip()) for p in date_line.split(',')]
    start_date = datetime(parts[0], parts[1], parts[2])
    return start_date

##--Store agglomerated data here: --##
AIMMS_dfs = []
 
##--Loop through each flight, pulling and analyzing data--##
for flight in flights_to_analyze:
    
    flight_number = flight

    ##--Populate flight_dir established in above function--##
    flight_dir = os.path.join(directory, flight)
    
    ##--Pull meteorological data from AIMMS monitoring system--##
    aimms_files = find_files(flight_dir, "AIMMS_POLAR6")
    if aimms_files:
        ##--Pull the file pathname--##
        aimms_file = aimms_files[0]
        ##--Pull the dataset itself--##
        aimms = icartt.Dataset(aimms_file)
        
        ##--Pull date from header--##
        date = get_icartt_dates(aimms_file)

    ##--AIMMS Data--##
    altitude = aimms.data['Alt'] # in mamsl
    latitude = aimms.data['Lat'] # in degrees
    longitude = aimms.data['Lon'] # in degrees
    temperature = aimms.data['Temp'] + 273.15 # in K
    pressure = aimms.data['BP'] # in pa
    aimms_time = aimms.data['TimeWave'] # seconds since midnight
    
    ##--Create a variable called 'flight' with date of flight--##
    flight_date = date.date()
    
    ##--Convert flight into a string--##
    flight_date_str = flight_date.strftime("%Y%m%d")  #YYYYMMDD format
    
    ##--Create a series of date times--##
    aimms_datetime = pd.to_datetime(aimms_time, unit='s', origin=flight_date)
    
    ##--Put all AIMMS data into a dataframe--##
    df = pd.DataFrame({'Flight_date':flight_date_str, 'Flight_num': flight_number, 
                       'datetime': aimms_datetime, 'Time_start':aimms_time,
                       'Alt': altitude, 'Lat': latitude, 'Lon': longitude})

    ##--Pull CPC files--##
    CPC10_files = find_files(flight_dir, 'CPC3772')
    CPC3_files = find_files(flight_dir, 'CPC3776')
 
    if CPC10_files and CPC3_files:
        CPC10 = icartt.Dataset(CPC10_files[0])
        CPC3 = icartt.Dataset(CPC3_files[0])
    else:
        print(f"Missing CPC data for {flight}. Skipping...")
        continue
    
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
        
    df['ptemp'] = potential_temp
    
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
    nuc_particles = (CPC_df['CPC3_conc'] - CPC_df['CPC10_conc'])

    ##--Change calculated particle counts less than zero to NaN--##
    nuc_particles = np.where(nuc_particles >= 0, nuc_particles, np.nan)


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
    
    ##--Convert to an array--##
    nuc_error_3sigma = nuc_error_3sigma.to_numpy()
    
    ##--Subtract error from nucleating particles--##f
    ##--First condition, then outcome, then the 'else' outcome--##
    nuc_significant = np.where(nuc_particles > nuc_error_3sigma, nuc_particles, np.nan)

    df['nuc_significant'] = nuc_significant
    
    ##--Agglomerate dataframes into a list--##
    AIMMS_dfs.append(df)
    

##--Concatenate the list of dataframes into one large df--##
Netcare = pd.concat(AIMMS_dfs, ignore_index=True)

##--Write and save parquet file--##
#Netcare.to_parquet(r"C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data\raw\Netcare.parquet", engine='pyarrow')

##--Write and save csv file--##
Netcare.to_csv(r'C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data\raw\Netcare.csv', index=False)

    
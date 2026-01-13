# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 08:10:45 2025

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
directory = r"C:\Users\repooley\REP_PhD\Arctic_NPF\FIREACE1998\data\raw"

##--Select flights to analyze (Flight1 thru Flight18)--##
flights_to_analyze = ["Flight3", 
                      'Flight7', 'Flight8', 'Flight9', 'Flight10', 'Flight11',
                      'Flight13', 'Flight14', 'Flight15',
                      'Flight16', 'Flight17', 'Flight18']

##--PCASP bins--##
bins_filepath = r"C:\Users\repooley\REP_PhD\Arctic_NPF\FIREACE1998\data\raw\FIREACE1998_PCASP_bins.csv"

################################
##--Open Files and Pull Data--##
################################

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


##--Store agglomerated data here: --##
dfs = []
 
##--Loop through each flight, pulling and analyzing data--##
for flight in flights_to_analyze:
    
    flight_number = flight
    
    ##--Pull csv file containing all data--##
    files = find_files(directory, flight, "FIREACE")

    ##--The non-averaged data is always the first file--##
    if files:
        data = pd.read_csv(files[0])
        
        ##--Extract the date from the filename--##
        filename = os.path.basename(files[0])
        date_str = filename[-12:-4]  # eg in format "19980409"
        date = pd.to_datetime(date_str, format="%Y%m%d")  

    ##--Pull data variables from file--##
    time = data['Time'] # HHMMSS UTC time
    flight_date = date.date()
    
    pressure = data['Pressure'] * 100 # in Pa
    temperature = data['Temperature'] + 273.15 # in K
    RH = data['RH'] # percent wrt water
    altitude = data['Altitude'] # in m (agl?)
    latitude = data['Latitude'] # degrees
    
    ##--Make sure longitude is in degrees WEST--##
    longitude = data['Longitude']*-1 # degrees
    
    ##--Convert flight into a string--##
    flight_date_str = flight_date.strftime("%Y%m%d")  #YYYYMMDD format
    
    ##--Create a series of date times--##
    flight_datetime = pd.to_datetime(time, unit='s', origin=flight_date)
    
    ##--Put data into a dataframe--##
    df = pd.DataFrame({'Flight_date':flight_date_str, 'Flight_num': flight_number, 
                       'datetime': flight_datetime, 'Time_start':time,
                       'Alt': altitude, 'Lat': latitude, 'Lon': longitude,
                       'Temp': temperature, 'Pressure': pressure, 'RH': RH})
    
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

    ##########################
    ##--Pull particle data--##
    ##########################

    ##--Particle data, 3 and 10 nm cutoffs, respectively--##
    CPC3_data = data['CN3025'] # Uncorrected data has a flow issue - but corrected not populated for many flights
    CPC10_data = data['CN7610']

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

    #############################
    ##--Propagate uncertainty--##
    #############################

    ##--Calculated error is too high - this is the 75th percentile median uncertainty across NETCARE--##
    nuc_error_3sigma = 133.71 
  
    ##--Subtract error from nucleating particles--##f
    ##--First condition, then outcome, then the 'else' outcome--##
    nuc_significant = np.where(nuc_particles > nuc_error_3sigma, nuc_particles, np.nan)

    df['nuc_significant'] = nuc_significant

    dfs.append(df)

##--Concatenate the list of dataframes into one large df--##
FIREACE = pd.concat(dfs, ignore_index=True)

##--Write and save parquet file--##
FIREACE.to_parquet(r"C:\Users\repooley\REP_PhD\Arctic_NPF\FIREACE1998\data\raw\FIREACE.parquet", engine='pyarrow')

##--Write and save csv file--##
FIREACE.to_csv(r'C:\Users\repooley\REP_PhD\Arctic_NPF\FIREACE1998\data\raw\FIREACE.csv', index=False)

    
        
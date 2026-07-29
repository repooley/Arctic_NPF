# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 14:45:40 2026

@author: repooley
"""

import icartt
import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime
import pyarrow
import xarray as xr

###################
##--User inputs--##
###################
 
##--Set the base directory to project folder--##
directory = r"C:\Users\repooley\REP_PhD\Arctic_NPF\ATom2018\data\raw"

##--Select flights to analyze (Flight2, Flight10-12)--##
flights_to_analyze = ["Flight2", "Flight10", "Flight11", "Flight12"]

################################
##--Open Files and Pull Data--##
################################

##--Define function that creates datasets from filenames--##
def find_files(directory, flight, partial_name):
    ##--flight data are stored in a folder called "raw"--##
    flight_dir = os.path.join(directory, flight)
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

##--Store processed data here: --##
dfs = []
dfs_significant = []

##--Loop through each flight in the list--##
for flight in flights_to_analyze:
    
    #########################
    ##--Open ICARTT Files--##
    #########################
    
    files = find_files(directory, flight, '.ict')

    dataset = icartt.Dataset(find_files(directory, flight, '.ict')[0])
    
    if files:
        ##--Pull the file pathname--##
        file = files[0]
        
        ##--Pull date from header--##
        date = get_icartt_dates(file)

    flight_date = date.date()
    
    ##--Make sure flight date is a string so xarray can read it later--##
    flight_date = str(flight_date)
    
    #################
    ##--Pull data--##
    #################
    
    altitude = dataset.data['G_ALT'] # in m (The G_ variables are the best for aircraft position)
    latitude = dataset.data['G_LAT'] # deg
    longitude = dataset.data['G_LONG'] # deg
    temperature = dataset.data['T'] # in K
    pressure = dataset.data['P'] * 100 # in Pa
    RH = dataset.data['Relative_Humidity'] # wrt water, percent
    time =dataset.data['UTC_Start'] # seconds since midnight UTC
    nucleating = dataset.data['N_nucl_AMP'] # num/cm^3 STP (2.7-12 nm)
    aitken = dataset.data['N_aitken_AMP'] # num/cm^3 STP (12-60 nm)
    
    print(latitude)
    
    ##--Create a series of date times--##
    flight_datetime = pd.to_datetime(time, unit='s', origin=flight_date)

    ##--There are notable outliers in the nucleating data--##
    
    ##--First convert to a series for calc--##
    nucleating_series = pd.Series(nucleating)
    
    ##--REMOVE OUTLIERS above 99.9th percentile--##
    p = 0.999
    
    ##--Compute threshold for each UHSAS column--##
    nucleating_thresh = nucleating_series.quantile(p)
    
    ##--keep only rows where each bin is below its threshold--##
    nucleating_filtered = nucleating_series.mask(nucleating_series > nucleating_thresh)
    
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
    
    ##--Put all data into a dataframe--##
    df = pd.DataFrame({'Flight_date':flight_date, 'Flight_num': flight, 
                       'datetime': flight_datetime, 'Time_start':time,
                       'Alt': altitude, 'Lat': latitude, 'Lon': longitude,
                       'Temp': temperature, 'ptemp':potential_temp})
    
    ##--This is the 75th percentile median uncertainty across NETCARE--##
    nuc_error_3sigma = 54
    
    ##--Subtract error from nucleating particles--##f
    ##--First condition, then outcome, then the 'else' outcome--##
    nuc_significant = np.where(nucleating_filtered > nuc_error_3sigma, nucleating_filtered, np.nan)

    df['nuc_significant'] = nuc_significant
    
    ##--Drop rows where the potential temperature is above 310 K for comparison to other campaigns--##
    df = df.mask(df['ptemp']>310)
    
    ##--Constrain the region to the high Arctic--##
    df = df.mask(df['Lat']<66.5) 
    
    df_significant = df[df['nuc_significant'].notna()]
    
    dfs.append(df)
    
    dfs_significant.append(df_significant)
    
##--Concatenate the list of dataframes into one large df--##
ATom = pd.concat(dfs, ignore_index=True)

ATom_significant = pd.concat(dfs_significant, ignore_index=True)

##--Write netCDF file--##
ds = xr.Dataset.from_dataframe(ATom)

##--Write and save parquet file--##
ds.to_netcdf(r"C:\Users\repooley\REP_PhD\Arctic_NPF\ATom2018\data\raw\ATom.nc")

##--Write and save csv file--##
ATom.to_csv(r'C:\Users\repooley\REP_PhD\Arctic_NPF\ATom2018\data\raw\ATom.csv', index=False)

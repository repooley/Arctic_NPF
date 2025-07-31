# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 09:06:44 2025

@author: repooley
"""

import icartt
import os
import glob
import pandas as pd
from datetime import datetime

###################
##--User inputs--##
###################
 
##--Set the base directory to project folder--##
directory = r"C:\Users\repooley\REP_PhD\NETCARE2015\data\raw"

##--Choose which flights to analyze here!--##
flights_to_analyze = ["Flight2", "Flight3", "Flight4", "Flight5", "Flight6", "Flight7", "Flight8", "Flight9", "Flight10"]

#######################################
##--Open ICARTT Files and Pull Data--##
#######################################
 
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
    aimms_time = aimms.data['TimeWave'] # seconds since midnight
    
    ##--Create a variable called 'flight' with date of flight--##
    flight_date = date.date()
    
    ##--Create a series of date times--##
    aimms_datetime = pd.to_datetime(aimms_time, unit='s', origin=flight_date)
    
    ##--Convert flight into a string--##
    flight = flight_date.strftime("%Y%m%d")  #YYYYMMDD format
    
    ##--Put all data into a dataframe--##
    df = pd.DataFrame({'Flight':flight, 'datetime': aimms_datetime, 'Time_start':aimms_time,
                       'Alt': altitude, 'Lat': latitude, 'Lon': longitude})
    
    ##--Agglomerate dataframes into a list--##
    AIMMS_dfs.append(df)
    
##--Concatenate the list of dataframes into one large df--##
Netcare = pd.concat(AIMMS_dfs, ignore_index=True)

##--Write and save parquet file--##
Netcare.to_parquet(r"C:\Users\repooley\REP_PhD\NETCARE2015\data\raw\Netcare.parquet", engine='pyarrow')

    
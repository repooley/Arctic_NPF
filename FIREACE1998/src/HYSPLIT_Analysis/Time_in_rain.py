# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 11:31:17 2026

@author: repooley
"""

import os
import numpy as np
import statistics 
import pandas as pd
from datetime import date

###################
##--User inputs--##
###################

##--Select flight (Flight1 thru Flight18)--##
flight = "Flight15" 

hysplit = r"C:\Users\repooley\REP_PhD\Arctic_NPF\FIREACE1998\data\raw\HYSPLIT\data\trajectories\5min_averaged"

##--Base output path for figures in directory--##
#output_path = r"C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data\processed\HYSPLIT"

##--Flights to analyze - flights 1-18--##
flights_to_analyze = ["Flight9", "Flight10", "Flight15"]

FIREACE = pd.read_csv(r"C:\Users\repooley\REP_PhD\Arctic_NPF\FIREACE1998\data\raw\FIREACE.csv")

##################
##--Pull Files--##
##################

##--Define function that finds all flights available--##
##--Create directory based on selected flight--##
def find_flights(directory, flight):
    flight_dir = os.path.join(hysplit, flight)
    return flight_dir

for flight in flights_to_analyze:
    
    ####################################
    ##--Assign date to flight number--##
    ####################################

    if flight=="Flight1":
        flight_date = date(2015, 4, 5)
    elif flight=="Flight2":
        flight_date = date(2015, 4, 7)
    elif flight=="Flight3" or flight=="Flight4":
        flight_date = date(2015, 4, 8)
    elif flight=="Flight5":
        flight_date = date(2015, 4, 9)
    elif flight=="Flight6":
        flight_date = date(2015, 4, 11)
    elif flight=="Flight7": 
        flight_date = date(2015, 4, 13)
    elif flight=="Flight8" or flight=="Flight9": 
        flight_date = date(2015, 4, 20)
    elif flight=="Flight10": 
        flight_date = date(2015, 4, 21)

    flight_directory = find_flights(hysplit, flight)
    
    ##--Get timestamps where trajectories were initialized--##
    ##--Trajectories were initialized every 10 minutes from the Netcare file--##
    single_flight = FIREACE[FIREACE['Flight_num'] == flight]
    
    start_utc = int(single_flight['Time_start'].min())
    end_utc = int(single_flight['Time_start'].max())
    UTCs = list(range(start_utc, end_utc +1, 300))
    
    ##--Subset Netcare to times in UTCs--##
    netcare_subset = single_flight[single_flight['Time_start'].isin(UTCs)]
    
    ##########################
    ##--Group trajectories--##
    ##########################
    
    ##--Sort trajectory outputs into signficant or non-significant NPF lists--##
    rain_hours_sig, rain_hours_nonsig = [], []
    
    for file, row in zip(sorted(os.listdir(flight_directory)), netcare_subset.itertuples(index=False)):
        
        ##--Determine which axis to use (NPF vs non-NPF)--##
        is_significant = pd.notna(row.nuc_significant)
            
        ##--\s denotes any whitespace character, + indicates one or more spaces--##
        df = pd.read_csv(os.path.join(hysplit, flight, file), sep=r'\s+')
        
        ##--Rename DATE to DAY--##
        df = df.rename(columns={'DATE': 'DAY'})
        
        ##--Change year to four digits, .apply() takes a function as an argument--##
        ##--A lambda function is local only--##
        df['YEAR'] = df['YEAR'].apply(lambda y: y + 2000)
        
        ##--Format for year, month, day, hour--##
        df['DateTime'] = pd.to_datetime({'year': df['YEAR'], 'month': df['MONTH'],
                'day': df['DAY'], 'hour': df['HOUR']})
      
        ##--Group by TRAJ to place each trajectory in time order--##
        for traj_num, group in df.groupby('TRAJ'):
            
            group = group.sort_values('DateTime')
            
            ##--Suggestion from GPT5 - deal with HYSPLIT wrapping around meridian--##
            # Normalize to -180 to 180 range
            group['LONG'] = ((group['LONG'] + 180) % 360) - 180
            
            lon = group['LONG'].values
            lat = group['LAT'].values
            altitudes = group['ALTITUDE'].values
            temps = group['AIR_TEMP'].tolist() 
            RHs = group['RELHUMID'].values
            rain = group['RAINFALL'].values
            
            ##--Compute relative time in days (backward from initialization)--##
            t0 = group['DateTime'].iloc[-1]
            time_rel = (group['DateTime'] - t0).dt.total_seconds() / 86400.0
            time_rel = time_rel.values  # ensure numpy array
        
            ##--Detect jumps >180° and break line by inserting NaNs--##
            jump_indices = np.where(np.abs(np.diff(lon)) > 180)[0]
            if len(jump_indices) > 0:
                for j in jump_indices[::-1]:  # reverse order to avoid index shift
                    lon = np.insert(lon, j + 1, np.nan)
                    lat = np.insert(lat, j + 1, np.nan)
                    altitudes = np.insert(altitudes, j + 1, np.nan)
                    time_rel = np.insert(time_rel, j + 1, np.nan)
                    temps = np.insert(temps, j + 1, np.nan)
                    RHs = np.insert(RHs, j + 1, np.nan)
                    rain = np.insert(rain, j + 1, np.nan)
    
            ##--Cut off trajectory within 1m of surface, HYSPLIT is iffy here--##
            if any(altitudes < 1):
                index_end = np.min(np.where(altitudes < 1))
            else:
                index_end = len(group) 
                
            ##--Establish threshold for rain--##
            rain_thresh = 0.5 #mm/hr
            
            if is_significant:
                
                rain_time_sig = rain[rain > rain_thresh]
                hours_in_rain_sig = len(rain_time_sig)
                
                rain_hours_sig.append(hours_in_rain_sig)
            else:
                
                rain_time_nonsig = rain[rain > rain_thresh]
                hours_in_rain_nonsig = len(rain_time_nonsig)
                
                rain_hours_nonsig.append(hours_in_rain_nonsig)
        
    
    if rain_hours_sig:                
        mean_time_in_rain_sig = statistics.mean(rain_hours_sig)
        print(f"{flight.replace('Flight', 'Flight ')} ({flight_date}) mean sig time in rain:", 
              mean_time_in_rain_sig, "hours")
    else:
        print(f"{flight.replace('Flight', 'Flight ')} ({flight_date}) no mean sig time in rain")
    
    mean_time_in_rain_nonsig = statistics.mean(rain_hours_nonsig)
    print(f"{flight.replace('Flight', 'Flight ')} ({flight_date}) mean nonsig time in rain:", 
          mean_time_in_rain_nonsig, "hours")  

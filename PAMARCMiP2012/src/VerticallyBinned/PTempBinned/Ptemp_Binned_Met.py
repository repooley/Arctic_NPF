# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 10:12:43 2026

@author: repooley
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from datetime import date
import icartt

##################
##--Open Files--##
##################

##--Set the base directory to project folder--##
directory = r"C:\Users\repooley\REP_PhD\Arctic_NPF\PAMARCMiP2012\data\raw"

CPC10_R1 = icartt.Dataset(r'C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data\raw\CPC_R1\CPC3772_Polar6_20150408_R1_L2.ict')

##--Select flight (Flight1 thru Flight9)--##
flights_to_analyze = ["Flight1", "Flight2", "Flight3", "Flight4", "Flight5", "Flight6", 
                      "Flight7", "Flight8", "Flight9"]

##--Define function that creates datasets from filenames--##
def find_files(directory, flight, partial_name):
    flight_dir = os.path.join(directory, flight)
    search_pattern = os.path.join(flight_dir, f"*{partial_name}*")
    return sorted(glob.glob(search_pattern))

for flight in flights_to_analyze:
    ##--Pull file--##
    data = pd.read_csv(find_files(directory, flight, ".csv")[0])
    
    #################
    ##--Pull data--##
    #################
    
    ##--Data--##
    altitude = data['Altitude'] # in m
    latitude = data['Latitude'] # in degrees
    temperature = data['Temp'] + 273.15 # in K
    pressure = data['Pressure'] # in pa
    time = data['Time'] # seconds since midnight
    RH = data['RH'] # percent wrt water
    
    df = pd.DataFrame({'temp': temperature, 'pressure': pressure, 'RH': RH})
    
    ###########################
    ##--Calc potential temp--##
    ###########################
    
    ##--Convert absolute temperature to potential temperature--##
    ##--Constants--##
    p_0 = 1E5 # Reference pressure in Pa (1000 hPa)
    k = 0.286 # Poisson constant for dry air
    
    ##--Generate empty list for potential temperature output--##
    potential_temp = []
    
    ##--Calculate potential temperature from ambient temp & pressure--##
    for T, P in zip(temperature, pressure):
        p_t = T*(p_0/P)**k
        potential_temp.append(p_t)
        
    PTemp_series = pd.Series(potential_temp)
    
    df['Ptemp'] = PTemp_series
    
    ####################################
    ##--Assign date to flight number--##
    ####################################
    
    if flight=="Flight1":
        flight_date = date(2012, 3, 29)
    elif flight=="Flight2":
        flight_date = date(2012, 3, 30)
    elif flight=="Flight3":
        flight_date = date(2012, 4, 2)
    elif flight=="Flight4" or flight=="Flight5":
        flight_date = date(2012, 4, 3)
    elif flight=="Flight6":
        flight_date = date(2012, 4, 4)
    elif flight=="Flight7": 
        flight_date = date(2012, 4, 5)
    elif flight=="Flight8": 
        flight_date = date(2012, 4, 6)
    elif flight=="Flight9": 
        flight_date = date(2012, 4, 7)

    ###############
    ##--BINNING--##
    ###############
          
    ##--Define number of bins here--##
    num_bins = 124
    
    ##--Compute the minimum and maximum ptemp, ignoring NaNs--##
    min_ptemp = df['Ptemp'].min(skipna=True)
    max_ptemp = df['Ptemp'].max(skipna=True)
    
    ##--Create bin edges from min_alt to max_alt--##
    bin_edges = np.linspace(min_ptemp, max_ptemp, num_bins + 1)
    
    ##--Pandas 'cut' splits altitude data into specified number of bins--##
    df['Ptemp_bin'] = pd.cut(df['Ptemp'], bins=bin_edges)
    
    ##--Group variables into each altitude bin--## 
    ##--Observed=false shows all bins, even empty ones--##
    binned_df = df.groupby('Ptemp_bin', observed=False).agg(
        
       ##--Aggregate data by mean, min, and max--##
        Ptemp_center=('Ptemp', 'median'), 
        temp_center=('temp', 'median'),
        temp_min=('temp', 'min'),
        temp_max=('temp', 'max'),
        temp_25th = ('temp', lambda x: x.quantile(0.25)),
        temp_75th = ('temp', lambda x: x.quantile(0.75)),
        RH_center=('RH', 'median'), 
        RH_min=('RH', 'min'),
        RH_max=('RH', 'max'),
        RH_25th=('RH', lambda x: x.quantile(0.25)),
        RH_75th=('RH', lambda x: x.quantile(0.75))
        
        ##--Reset the index so Altitude_bin is just a column--##
    ).reset_index()
    
    #%%
    ################
    ##--PLOTTING--##
    ################
    
    ##--Creates figure with 4 horizontally stacked subplots sharing a y-axis--##
    fig, axs = plt.subplots(1, 2, figsize=(6, 6), sharey=True)
    
    ##--First subplot: 60+ nm Particles vs Altitude--##
    
    ##--Averaged data in each bin is plotted against bin center--##
    axs[0].plot(binned_df['temp_center'], binned_df['Ptemp_center'], color='maroon')
    ##--Range is given by filling between data minimum and maximum for each bin--##
    axs[0].fill_betweenx(binned_df['Ptemp_center'], binned_df['temp_min'], 
                         binned_df['temp_max'], color='indianred', alpha=0.25)
    axs[0].fill_betweenx(binned_df['Ptemp_center'], binned_df['temp_25th'],
                        binned_df['temp_75th'], color='indianred', alpha=0.7)
    axs[0].set_ylabel('Potential Temperature (K)', fontsize=12)
    axs[0].set_xlabel('Temperature (K)')
    axs[0].set_title('Temperature')
    #axs[0].set_xlim(-50, 1500)
    
    ##--Second subplot: 10+ nm Particles vs Altitude--##
    axs[1].plot(binned_df['RH_center'], binned_df['Ptemp_center'], color='saddlebrown')
    axs[1].fill_betweenx(binned_df['Ptemp_center'], binned_df['RH_min'], 
                         binned_df['RH_max'], color='sandybrown', alpha=0.25)
    axs[1].fill_betweenx(binned_df['Ptemp_center'], binned_df['RH_25th'],
                        binned_df['RH_75th'], color='sandybrown', alpha=1)
    axs[1].set_title('Relative Humidity')
    axs[1].set_xlabel('RH (%)')
    #axs[1].set_xlim(-50, 2000)

    ##--Use f-string to embed flight # variable in plot title--##
    plt.suptitle(f"Meteorological Profiles - {flight.replace('Flight', 'Flight ')} ({flight_date})", fontsize=16)
    
    ##--Adjusts layout to prevent overlapping--## 
    plt.tight_layout(rect=[0, -0.02, 1, 0.99])
    
    ##--Use f-string to save file with flight# appended--##
    #output_path = f"{output_path}\\CPC_Data_{flight}"
    #plt.savefig(output_path, dpi=600, bbox_inches='tight') 
    
    plt.show()
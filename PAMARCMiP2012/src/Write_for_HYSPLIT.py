# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 09:26:42 2026

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
directory = r"C:\Users\repooley\REP_PhD\Arctic_NPF\PAMARCMiP2012\data\raw"

CPC10_R1 = icartt.Dataset(r'C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data\raw\CPC_R1\CPC3772_Polar6_20150408_R1_L2.ict')

##--Select flights to analyze (Flight1 thru Flight9)--##
flights_to_analyze = ["Flight1", "Flight2", "Flight3", "Flight6", 
                      "Flight7", "Flight8", "Flight9"]

################################
##--Open Files and Pull Data--##
################################

##--Define function that creates datasets from filenames--##
def find_files(directory, flight, partial_name):
    ##--flight data are stored in a folder called "raw"--##
    flight_dir = os.path.join(directory, flight)
    search_pattern = os.path.join(flight_dir, f"*{partial_name}*")
    return sorted(glob.glob(search_pattern))

##--Store processed data here: --##
dfs = []

for flight in flights_to_analyze:
    ##--Pull file--##
    data = pd.read_csv(find_files(directory, flight, ".csv")[0])
    
    flight_number = flight
    
    ##--Pull csv file containing all data--##
    files = find_files(directory, flight, "PA")

    ##--Always the first file--##
    if files:
        
        ##--Extract the date from the filename--##
        filename = os.path.basename(files[0])
        date_str = filename[2:8]  # eg in format "120330"
        date = pd.to_datetime(date_str, format="%y%m%d")  

    ##--Pull data variables from file--##
    time = data['Time'] # HHMMSS UTC time
    flight_date = date.date()
    
    pressure = data['Pressure'] * 100 # in Pa
    temperature = data['Temp'] + 273.15 # in K
    RH = data['RH'] # percent wrt water
    altitude = data['Altitude'] # in m (agl?)
    altitude = altitude.astype(float) # make sure not a series of int
    latitude = data['Latitude'] # degrees
    
    ##--Make sure longitude is in degrees WEST--##
    longitude = data['Longitude']*-1 # degrees
    
    ##--Convert flight into a string--##
    flight_date_str = flight_date.strftime("%y%m%d")  #YYMMDD format
    
    ##--Create a variable called 'flight' with date of flight--##
    flight_date = date.date()
    
    ##--Convert flight into a string--##
    flight_date_str = flight_date.strftime("%Y%m%d")  #YYYYMMDD format
    
    ##--Create a series of date times--##
    flight_datetime = pd.to_datetime(time, unit='s', origin=flight_date)
    
    ##--Put all data into a dataframe--##
    df = pd.DataFrame({'Flight_date':flight_date_str, 'Flight_num': flight_number, 
                       'datetime': flight_datetime, 'Time_start':time,
                       'Alt': altitude, 'Lat': latitude, 'Lon': longitude,
                       'Temp': temperature, 'Pressure': pressure})

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
    
    df['ptemp'] = PTemp_series
    
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
    
    df['n_10_60'] = n_10_60
    
    #############################
    ##--Calculate Uncertainty--##        
    #############################
    
    ##--Pull CPC data from R1 data--##
    CPC10_R1_conc = CPC10_R1.data['conc']
    
    ##--Isolate zero periods, setting conservative upper limit of 50 counts--##
    ##--Numpy doesn't recognize -9999 as NaN, tell it to ignore these values--##
    CPC10_zeros_c = CPC10_R1_conc[(CPC10_R1_conc < 50) & (CPC10_R1_conc != -99999)]
    
    ##--Calculate standard deviation of zeros--##
    # Use ddof=1 for sample standard deviation
    CPC10_sigma = np.std(CPC10_zeros_c, ddof=1)
    
    greater10nm_error = 3*CPC10_sigma
    
    ##--This is the 75th percentile median uncertainty across NETCARE--##
    nuc_error_3sigma = 133.71 
    
    ##--UHSAS doesn't have zero periods, using Poisson counting uncertainty--##
    UHSAS_total_sqrt = np.sqrt(UHSAS_denorm_counts)
    
    ##--Use simple sum of UHSAS uncertainties per bin for conservative estimate--##
    ##--Similar result as using sqrt of squares but erring on side of caution--##
    UHSAS_total_error = UHSAS_total_sqrt.sum(axis=1)
    
    ##--Calculate error in difference between CPC10 and UHSAS + OPC--##
    aitken_error_3sigma = (((greater10nm_error)**2 + (UHSAS_total_error)**2)**(0.5))*3
    
    aitken_significant = np.where(n_10_60 > aitken_error_3sigma, n_10_60, np.nan)
    
    df['aitken_significant'] = aitken_significant
    
    dfs.append(df)
    
##--Concatenate the list of dataframes into one large df--##
PAMARCMiP = pd.concat(dfs, ignore_index=True)

##--Write netCDF file--##
ds = xr.Dataset.from_dataframe(PAMARCMiP)

##--Write and save parquet file--##
ds.to_netcdf(r"C:\Users\repooley\REP_PhD\Arctic_NPF\PAMARCMiP2012\data\raw\PAMARCMiP.nc")

##--Write and save csv file--##
PAMARCMiP.to_csv(r'C:\Users\repooley\REP_PhD\Arctic_NPF\PAMARCMiP2012\data\raw\PAMARCMiP.csv', index=False)

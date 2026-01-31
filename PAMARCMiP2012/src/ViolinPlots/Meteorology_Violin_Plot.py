# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 10:20:28 2026

@author: repooley
"""

import icartt
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date

###################
##--User inputs--##
###################

##--Set the base directory to project folder--##
directory = r"C:\Users\repooley\REP_PhD\Arctic_NPF\PAMARCMiP2012\data"

##--Select flight (Flight1 thru Flight9)--##
flights_to_analyze = ["Flight1", "Flight2", "Flight3", "Flight6", 
                      "Flight7", "Flight8", "Flight9"]

#########################
##--Open ICARTT Files--##
#########################

##--Define function that creates datasets from filenames--##
def find_files(directory, flight, partial_name):
    ##--flight data are stored in a folder called "raw"--##
    flight_dir = os.path.join(directory, "raw", flight)
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
    RH = data['RH'] # % 
    
    ##--The first datapoint in 'latitude' column is erraneous (47.12 N)--##
    latitude = latitude.where(latitude >= 50, np.nan)
    
    temp_RH_df = pd.DataFrame({'Temperature': temperature, 'RH': RH, 'Time':time}).set_index('Time')
    
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
    n_10_60_df = pd.DataFrame({'35': n_10_60, 'time':time}).set_index('time')
    
    #####################################
    ##--Calculate std dev from zeroes--##        
    #####################################
    
    ##--Pull datasets with zeros not filtered out--##
    CPC10_R1 = icartt.Dataset(r'C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data\raw\CPC_R1\CPC3772_Polar6_20150408_R1_L2.ict')
    
    ##--Pull CPC data from R1 data--##
    CPC10_R1_conc = CPC10_R1.data['conc']
    
    ##--Isolate zero periods, setting conservative upper limit of 50 counts--##
    ##--Numpy doesn't recognize -9999 as NaN, tell it to ignore these values--##
    CPC10_zeros_c = CPC10_R1_conc[(CPC10_R1_conc < 50) & (CPC10_R1_conc != -99999)]
    
    ##--Calculate standard deviation of zeros--##
    # Use ddof=1 for sample standard deviation
    CPC10_sigma = np.std(CPC10_zeros_c, ddof=1)
    
    greater10nm_error = 3*CPC10_sigma
    
    ##--UHSAS doesn't have zero periods, using Poisson counting uncertainty--##
    UHSAS_total_sqrt = np.sqrt(UHSAS_denorm_counts)
    
    ##--Use simple sum of UHSAS uncertainties per bin for conservative estimate--##
    ##--Similar result as using sqrt of squares but erring on side of caution--##
    UHSAS_total_error = UHSAS_total_sqrt.sum(axis=1)
    
    ##--Calculate error in difference between CPC10 and UHSAS + OPC--##
    aitken_error_3sigma = (((greater10nm_error)**2 + (UHSAS_total_error)**2)**(0.5))*3
    
    aitken_error_3sigma = pd.DataFrame({'aitken_error_3sigma': aitken_error_3sigma, 'time': time}).set_index('time')
    
    #############################################
    ##--Filter to Aitken and non-Aitken times--##
    #############################################
    
    temperature_n_10_60 = pd.DataFrame({'Temperature': temp_RH_df['Temperature'], 'Aitken': n_10_60_df['35'],
                                     'LoD': aitken_error_3sigma['aitken_error_3sigma']})
    
    temperature_aitken = temperature_n_10_60['Temperature'][temperature_n_10_60['Aitken'] > temperature_n_10_60['LoD']]
    temperature_noaitken = temperature_n_10_60['Temperature'][temperature_n_10_60['Aitken'] <= temperature_n_10_60['LoD']]
    temperature_df = {'Aitken': temperature_aitken, 'No Aitken': temperature_noaitken}
    
    RH_n_10_60 = pd.DataFrame({'RH': temp_RH_df['RH'], 'Aitken': n_10_60_df['35'],
                               'LoD': aitken_error_3sigma['aitken_error_3sigma']})
    RH_aitken = RH_n_10_60['RH'][RH_n_10_60['Aitken'] > RH_n_10_60['LoD']]
    RH_noaitken = RH_n_10_60['RH'][RH_n_10_60['Aitken'] <= RH_n_10_60['LoD']]
    
    RH_df = {'Aitken': RH_aitken, 'No Aitken': RH_noaitken}
    
    #############
    ##--Stats--##
    #############
    
    ##--Counts--##
    temperature_aitken_count = len(temperature_aitken)
    temperature_noaitken_count = len(temperature_noaitken)
    
    RH_aitken_count = len(RH_aitken)
    RH_noaitken_count = len(RH_noaitken)

    ################
    ##--Plotting--##
    ################
    
    ##--Assign pallettes--##
    palette = {'Aitken':'#2f6794', 'No Aitken':'#1e537e'}

    ##--Temperature--##
    fig, ax = plt.subplots(figsize = (4,6))
    ##--Cut=0 disallows interpolation beyond the data extremes--##
    temperature_plot = sns.violinplot(data=temperature_df, palette=palette,
                                       inner_kws={'whis_width': 0, 'solid_capstyle':'butt'}, ax=ax, cut=0)
    ax.set(xlabel='')
    ax.set(ylabel='Temperature (K)')
     
    ax.set(title=f"Temperature - PAMARCMiP {flight.replace('Flight', 'Flight ')} ({flight_date})")
        
    ##--Add text labels with N--##
    plt.text(0.25, 0.12, "N={}".format(temperature_aitken_count), transform=fig.transFigure, fontsize=10, color='dimgrey')
    plt.text(0.63, 0.12, "N={}".format(temperature_noaitken_count), transform=fig.transFigure, fontsize=10, color='dimgrey')

    #plt.savefig(f"{output_path}\\Sinks\condensation\conden_{flight}", dpi=600)
    
    plt.show()
    
    ##--RH--##
    fig, ax = plt.subplots(figsize = (4,6))
    ##--Cut=0 disallows interpolation beyond the data extremes--##
    RH_plot = sns.violinplot(data=RH_df, palette=palette,
                                       inner_kws={'whis_width': 0, 'solid_capstyle':'butt'}, ax=ax, cut=0)
    ax.set(xlabel='')
    ax.set(ylabel='Relative Humidity w.r.t. Water (%)')
     
    ax.set(title=f"Relative Humidity - PAMARCMiP {flight.replace('Flight', 'Flight ')} ({flight_date})")
        
    ##--Add text labels with N--##
    plt.text(0.25, 0.12, "N={}".format(RH_aitken_count), transform=fig.transFigure, fontsize=10, color='dimgrey')
    plt.text(0.63, 0.12, "N={}".format(RH_noaitken_count), transform=fig.transFigure, fontsize=10, color='dimgrey')

    #plt.savefig(f"{output_path}\\Sinks\condensation\conden_{flight}", dpi=600)
    
    plt.show()
    
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 10:04:30 2026

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
flights_to_analyze = ["Flight1", "Flight2", "Flight3", "Flight6", 
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
    n_10_60 = pd.DataFrame({'35.5': n_10_60, 'time':time})
    
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
    
    ##--Creates a Pandas dataframe for particle data--##
    df = pd.DataFrame({'Ptemp': PTemp_series, 'CPC10_conc': CPC10_filtered, 'UHSAS_total': UHSAS_total['Total_count'],
                       'n_10_60': n_10_60['35.5'], 'aitken_error_3sigma': aitken_error_3sigma})
    
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
        UHSAS_total_center=('UHSAS_total', 'median'),
        UHSAS_total_min=('UHSAS_total', 'min'),
        UHSAS_total_max=('UHSAS_total', 'max'),
        UHSAS_total_25th = ('UHSAS_total', lambda x: x.quantile(0.25)),
        UHSAS_total_75th = ('UHSAS_total', lambda x: x.quantile(0.75)),
        CPC10_conc_center=('CPC10_conc', 'median'), 
        CPC10_conc_min=('CPC10_conc', 'min'),
        CPC10_conc_max=('CPC10_conc', 'max'),
        CPC10_conc_25th=('CPC10_conc', lambda x: x.quantile(0.25)),
        CPC10_conc_75th=('CPC10_conc', lambda x: x.quantile(0.75)),
        n_10_60_center=('n_10_60', 'median'), 
        n_10_60_min=('n_10_60', 'min'),
        n_10_60_max=('n_10_60', 'max'), 
        n_10_60_25th=('n_10_60', lambda x: x.quantile(0.25)),
        n_10_60_75th=('n_10_60', lambda x: x.quantile(0.75)),
      
        ##--And Aitken mode (10-85 nm) particles--##
        aitken_error_center=('aitken_error_3sigma', 'median')
        
        ##--Reset the index so Altitude_bin is just a column--##
    ).reset_index()
    
    #%%
    ################
    ##--PLOTTING--##
    ################
    
    ##--Creates figure with 4 horizontally stacked subplots sharing a y-axis--##
    fig, axs = plt.subplots(1, 3, figsize=(9, 6), sharey=True)
    
    ##--First subplot: 60+ nm Particles vs Altitude--##
    
    ##--Averaged data in each bin is plotted against bin center--##
    axs[0].plot(binned_df['UHSAS_total_center'], binned_df['Ptemp_center'], color='maroon')
    ##--Range is given by filling between data minimum and maximum for each bin--##
    axs[0].fill_betweenx(binned_df['Ptemp_center'], binned_df['UHSAS_total_min'], 
                         binned_df['UHSAS_total_max'], color='indianred', alpha=0.25)
    axs[0].fill_betweenx(binned_df['Ptemp_center'], binned_df['UHSAS_total_25th'],
                        binned_df['UHSAS_total_75th'], color='indianred', alpha=0.7)
    axs[0].set_ylabel('Potential Temperature (K)', fontsize=12)
    axs[0].set_xlabel('Counts/cm\u00b3')
    axs[0].set_title('N \u2265 60 nm')
    #axs[0].set_xlim(-50, 1500)
    
    ##--Second subplot: 10+ nm Particles vs Altitude--##
    axs[1].plot(binned_df['CPC10_conc_center'], binned_df['Ptemp_center'], color='saddlebrown')
    axs[1].fill_betweenx(binned_df['Ptemp_center'], binned_df['CPC10_conc_min'], 
                         binned_df['CPC10_conc_max'], color='sandybrown', alpha=0.25)
    axs[1].fill_betweenx(binned_df['Ptemp_center'], binned_df['CPC10_conc_25th'],
                        binned_df['CPC10_conc_75th'], color='sandybrown', alpha=1)
    axs[1].set_title('N \u2265 10 nm')
    axs[1].set_xlabel('Counts/cm\u00b3')
    #axs[1].set_xlim(-50, 2000)
    
    ##--Third subplot: 10-60 nm particles vs Altitude--##
    axs[2].plot(binned_df['n_10_60_center'], binned_df['Ptemp_center'], color='darkcyan')
    axs[2].fill_betweenx(binned_df['Ptemp_center'], binned_df['n_10_60_min'], 
                         binned_df['n_10_60_max'], color='turquoise', alpha=0.25)
    axs[2].fill_betweenx(binned_df['Ptemp_center'], binned_df['n_10_60_25th'],
                        binned_df['n_10_60_75th'], color='mediumturquoise', alpha=1)
    
    ##--Plot uncertainty as its own trace--##
    axs[2].plot(binned_df['aitken_error_center'], binned_df['Ptemp_center'], color='crimson', 
                linestyle='dashed', label='3$\sigma$ \nuncertainty')
    
    axs[2].legend(loc='lower right')
    
    ##--Subscript 10-60--##
    axs[2].set_title('$N_{10-60}$')
    axs[2].set_xlabel('Counts/cm\u00b3')
    #axs[2].set_xlim(-50, 2000)
    
    ##--Use f-string to embed flight # variable in plot title--##
    plt.suptitle(f"Vertical Particle Count Profiles - {flight.replace('Flight', 'Flight ')} ({flight_date})", fontsize=16)
    
    ##--Adjusts layout to prevent overlapping--## 
    plt.tight_layout(rect=[0, -0.02, 1, 0.99])
    
    ##--Use f-string to save file with flight# appended--##
    #output_path = f"{output_path}\\CPC_Data_{flight}"
    #plt.savefig(output_path, dpi=600, bbox_inches='tight') 
    
    plt.show()
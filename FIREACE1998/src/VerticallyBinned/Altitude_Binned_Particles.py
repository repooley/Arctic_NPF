# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 14:04:54 2025

@author: repooley
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from datetime import date

###################
##--User inputs--##
###################

##--Set the base directory to project folder--##
directory = r"C:\Users\repooley\REP_PhD\Arctic_NPF\FIREACE1998\data"

##--Flights to analyze - flights 1-18--##
flights_to_analyze = ["Flight3",  
                      "Flight7", "Flight8", "Flight9", "Flight10", "Flight11", "Flight12",
                      "Flight13", "Flight14", "Flight15", "Flight16", "Flight17", "Flight18"]

##--Base output path in directory--##
output_path = r"C:\Users\repooley\REP_PhD\Arctic_NPF\FIRACE1998\data\processed\VerticallyBinnedData"

PCASP_bins_path = r"C:\Users\repooley\REP_PhD\Arctic_NPF\FIREACE1998\data\raw\FIREACE1998_PCASP_bins.csv"

#%%

################################
##--Open Files and Pull Data--##
################################

##--Define function that creates datasets from filenames--##
def find_files(directory, flight, partial_name):
    ##--flight data are stored in a folder called "raw"--##
    flight_dir = os.path.join(directory, "raw", flight)
    search_pattern = os.path.join(flight_dir, f"*{partial_name}*")
    return sorted(glob.glob(search_pattern))

for flight in flights_to_analyze:

    ##--'raw' contains a 1hz and 2min datafile, the 1hz one is always first--##
    data = pd.read_csv(find_files(directory, flight, "FIREACE")[0])
    
    ##--Pull data variables from file--##
    time = data['Time'] # HHMMSS UTC time
    pressure = data['Pressure'] * 100 # in Pa
    temperature = data['Temperature'] + 273.15 # in K
    RH = data['RH'] # percent wrt water
    altitude = data['Altitude'] # in m (agl?)
    latitude = data['Latitude'] # degrees
    longitude = data['Longitude'] # degrees
    
    ##--Particle data, 3 and 10 nm cutoffs, respectively--##
    CPC3_data = data['CN3025_corrected'] # Uncorrected data has a flow issue
    CPC10_data = data['CN7610']
    
    averaged_data = pd.read_csv(find_files(directory, flight, "FIREACE")[1])
    averaged_data = averaged_data.set_index('Time', drop=False)

    PCASP_bins = pd.read_csv(PCASP_bins_path)

    PCASP_data = averaged_data.iloc[:, 14:29] # select PCASP data

    ##--Add time, total_num to UHSAS_bins df--##
    PCASP_data.insert(0, 'Time', averaged_data['Time'])

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
    
    ####################################
    ##--Assign date to flight number--##
    ####################################
    
    if flight=="Flight1":
        flight_date = date(1998, 4, 8)
    elif flight=="Flight2":
        flight_date = date(1998, 4, 9)
    elif flight=="Flight3":
        flight_date = date(1998, 4, 12)
    elif flight=="Flight4":
        flight_date = date(1998, 4, 14)
    elif flight=="Flight5":
        flight_date = date(1998, 4, 15)
    elif flight=="Flight6":
        flight_date = date(1998, 4, 16)
    elif flight=="Flight7" or flight=="Flight8": 
        flight_date = date(1998, 4, 17)
    elif flight=="Flight9": 
        flight_date = date(1998, 4, 18)
    elif flight=="Flight10" or flight=="Flight11": 
        flight_date = date(1998, 4, 21)
    elif flight=="Flight12":
        flight_date = date(1998, 4, 22)
    elif flight=="Flight13":
        flight_date = date(1998, 4, 24)
    elif flight=="Flight14":
        flight_date = date(1998, 4, 25)
    elif flight=="Flight15":
        flight_date = date(1998, 4, 27)
    elif flight=="Flight16" or flight=="Flight17": 
        flight_date = date(1998, 4, 28)
    elif flight=="Flight18": 
        flight_date = date(1998, 4, 29)
    
    #%%
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
    df = pd.DataFrame({'Altitude': altitude, 'CPC3_conc':CPC3_conc_STP, 'CPC10_conc': CPC10_conc_STP})
    
    ##--Calculate N3-10 particles--##
    nuc_particles = (df['CPC3_conc'] - df['CPC10_conc'])
    
    ##--Change calculated particle counts less than zero to NaN--##
    nuc_particles = np.where(nuc_particles >= 0, nuc_particles, np.nan)
    
    ##--Add nucleating particles to df--##
    df['nuc_particles'] = nuc_particles
    
    #########################
    ##--Averaged CPC Data--##
    #########################
    
    CPC_averaged_data = pd.DataFrame({'CPC3': averaged_data['CN3025'], 'CPC10': averaged_data['CN7610']}) 

    ##--Add time, total_num to UHSAS_bins df--##
    CPC_averaged_data.insert(0, 'Time', averaged_data['Time'])

    ##--Set time as the index for later alignment--##
    CPC_averaged_data = CPC_averaged_data.set_index('Time')
    
    ##--Calculate *averaged* nucleating particles--##
    n_3_10_averaged = (CPC_averaged_data['CPC3'] - CPC_averaged_data['CPC10'])
    
    ##--Change calculated particle counts less than zero to NaN--##
    n_3_10_averaged = np.where(n_3_10_averaged >= 0, n_3_10_averaged, np.nan)

    ##--Create empty list for n_3_10 particles--##
    n_3_10_averaged_STP = []

    for n_3_10, T, P in zip(n_3_10_averaged, averaged_data['Temperature']+273.15, averaged_data['Pressure']*100):
        if np.isnan(T) or np.isnan(P) or np.isnan(n_3_10):
            ##--Append with NaN if any input is NaN--##
            n_3_10_averaged_STP.append(np.nan)
        else:
            ##--Perform conversion if all inputs are valid--##
            corrected_n_3_10_averaged = n_3_10 * (P_STP / P) * (T / T_STP)
            n_3_10_averaged_STP.append(corrected_n_3_10_averaged)
                
    ##--Convert back to DataFrame with same columns and index--##
    n_3_10_averaged_STP = pd.DataFrame({'n_3_10_STP': n_3_10_averaged_STP}, index=CPC_averaged_data.index)
    
    ##--Make a separate dataframe for the averaged data--##
    df_averaged = pd.DataFrame({'Altitude': averaged_data['Altitude'], 
                                'Latitude': averaged_data['Latitude'], 
                                'Time': averaged_data['Time']})
    
    ##--Reindex df_averaged to time--##
    df_averaged = df_averaged.set_index('Time', drop=False)
    
    ##--Add PCASP data to the dataframe--##
    df_averaged = pd.concat([df_averaged, n_3_10_averaged_STP], axis=1)

    ######################
    ##--Calc N(10-130)--##
    ######################
    
    ##--Calculate the total in STP--##
    PCASP_total_STP = []
    
    for total, T, P in zip(averaged_data['PCTcon'], averaged_data['Temperature']+273.15, averaged_data['Pressure']*100):
        if np.isnan(T) or np.isnan(P) or np.isnan(total):
            ##--Append with NaN if any input is NaN--##
            PCASP_total_STP.append(np.nan)
        else:
            ##--Perform conversion if all inputs are valid--##
            corrected_total_averaged = total * (P_STP / P) * (T / T_STP)
            PCASP_total_STP.append(corrected_total_averaged)
    
    ##--Create df with UHSAS total counts--##
    PCASP_total = pd.DataFrame({'Time': averaged_data['Time'], 'Total_count': PCASP_total_STP})
    
    ##--Set time as the index for later alignment--##
    PCASP_total = PCASP_total.set_index('Time')
    
    ##--Add the PCASP total to the averaged df--##
    df_averaged = pd.concat([df_averaged, PCASP_total], axis=1)

    ##--Create df with CPC10 counts and set index to time--##
    CPC10_counts = pd.DataFrame({'Time':averaged_data['Time'], 'Counts':averaged_data['CN7610']}).set_index('Time')

    ##--Calculate particles below UHSAS lower cutoff--##
    n_10_130 = (averaged_data['CN7610'] - PCASP_total['Total_count'])

    ##--Change calculated particle counts less than zero to NaN--##
    n_10_130 = np.where(n_10_130 >= 0, n_10_130, np.nan)

    ##--Put N(10-130) bin center in a df--##
    n_10_130_center = pd.DataFrame([70])

    ##--Convert n_10_130 to a df--##
    n_10_130 = pd.DataFrame({'70': n_10_130, 'Time':averaged_data['Time']}).set_index('Time')
    
    df_averaged['grow_particles'] = n_10_130
    
    ##--Compute TOTAL counts from all size bins combined--##
    df_averaged['Total_particles_STP'] = (df_averaged['n_3_10_STP'].fillna(0) + 
    df_averaged['grow_particles'].fillna(0) + df_averaged['Total_count'].fillna(0))
    
    
    #%%
    #############################
    ##--Propagate uncertainty--##
    #############################

    ##--This is the 75th percentile median uncertainty across NETCARE--##
    nuc_error_3sigma = 133.71 
    
    df['nuc_error_3sigma'] = nuc_error_3sigma
    
    ##--PCASP uncertainty is quoted as +-5%--##
    PCASP_3sigma = 3*(0.05*(PCASP_total['Total_count']))
    
    ##--Not sure of what instruments were onboard Convair 580, using values from NETCARE instruments--##
    T_error = 0.3 # K, constant
    P_error = 100 + 0.0005*(averaged_data['Pressure'])
    
    PCASP_STP_3sigma = (PCASP_total['Total_count'])*(((P_error)/(averaged_data['Pressure']))**2 + 
            ((T_error)/(averaged_data['Temperature']))**2 + ((PCASP_3sigma)/(PCASP_total['Total_count'])))**(0.5)
    
    df_averaged['grow_error_3sigma'] = PCASP_STP_3sigma
    
    #%%
    ###############
    ##--BINNING--##
    ###############
    
    ##--Define number of bins here--##
    num_bins = 124
    
    ##--Assign outliers as NaN--##
    for col in df.columns:
        ##--Define outliers as in 99 or 1 percentile--##
        percentile_99 = df[col].quantile(0.99)
        percentile_01 = df[col].quantile(0.01)
        
        ##--Assign NaN--##
        df.loc[df[col] > percentile_99] = np.nan
        df.loc[df[col] < percentile_01] = np.nan
    
    ##--Compute the minimum and maximum altitude, ignoring NaNs--##
    min_alt = df['Altitude'].min(skipna=True)
    max_alt = df['Altitude'].max(skipna=True)
    
    ##--Create bin edges from min_alt to max_alt--##
    bin_edges = np.linspace(min_alt, max_alt, num_bins + 1)
    
    ##--Pandas 'cut' splits altitude data into specified number of bins--##
    df['Altitude_bin'] = pd.cut(df['Altitude'], bins=bin_edges)
    
    ##--Group variables into each altitude bin--## 
    ##--Observed=false shows all bins, even empty ones--##
    binned_df = df.groupby('Altitude_bin', observed=False).agg(
        
       ##--Aggregate data by mean, min, and max--##
        Altitude_center=('Altitude', 'median'), 
        CPC10_conc_center=('CPC10_conc', 'median'), 
        CPC10_conc_min=('CPC10_conc', 'min'),
        CPC10_conc_max=('CPC10_conc', 'max'),
        CPC10_conc_25th=('CPC10_conc', lambda x: x.quantile(0.25)),
        CPC10_conc_75th=('CPC10_conc', lambda x: x.quantile(0.75)),
        CPC3_conc_center=('CPC3_conc', 'median'), 
        CPC3_conc_min=('CPC3_conc', 'min'),
        CPC3_conc_max=('CPC3_conc', 'max'),
        CPC3_conc_25th=('CPC3_conc', lambda x: x.quantile(0.25)),
        CPC3_conc_75th=('CPC3_conc', lambda x: x.quantile(0.75)),
        nuc_particles_center=('nuc_particles', 'median'), 
        nuc_particles_min=('nuc_particles', 'min'),
        nuc_particles_max=('nuc_particles', 'max'), 
        nuc_particles_25th=('nuc_particles', lambda x: x.quantile(0.25)),
        nuc_particles_75th=('nuc_particles', lambda x: x.quantile(0.75)),
        
        ##--Bin the uncertainty of nucleating particles--##
        nuc_error_center=('nuc_error_3sigma', 'median')
        
        ##--Reset the index so Altitude_bin is just a column--##
    ).reset_index()
    
    ##--Pandas 'cut' splits altitude data into specified number of bins--##
    df_averaged['Altitude_bin'] = pd.cut(df_averaged['Altitude'], bins=bin_edges)
    
    binned_averaged_df = df_averaged.groupby('Altitude_bin', observed=False).agg(
        
        ##--Aggregate data by mean, min, and max--##
        Altitude_center=('Altitude', 'median'), 
        grow_particles_center=('grow_particles', 'median'),
        grow_particles_min=('grow_particles', 'min'),
        grow_particles_max=('grow_particles', 'min'),
        grow_particles_25th=('grow_particles', lambda x: x.quantile(0.25)),
        grow_particles_75th =('grow_particles', lambda x: x.quantile(0.75)),
        
        ##--And Aitken mode (grow) particles--##
        grow_error_center=('grow_error_3sigma', 'median')
        
    ).reset_index()
    
    #%%
    ################
    ##--PLOTTING--##
    ################
    
    ##--Creates figure with 3 horizontally stacked subplots sharing a y-axis--##
    fig, axs = plt.subplots(1, 4, figsize=(12, 6), sharey=True)
    
    ##--First subplot: 10+ nm Particles vs Altitude--##
    
    ##--Averaged data in each bin is plotted against bin center--##
    axs[0].plot(binned_df['CPC10_conc_center'], binned_df['Altitude_center'], color='maroon')
    ##--Range is given by filling between data minimum and maximum for each bin--##
    axs[0].fill_betweenx(binned_df['Altitude_center'], binned_df['CPC10_conc_min'], 
                         binned_df['CPC10_conc_max'], color='indianred', alpha=0.25)
    axs[0].fill_betweenx(binned_df['Altitude_center'], binned_df['CPC10_conc_25th'],
                        binned_df['CPC10_conc_75th'], color='indianred', alpha=0.7)
    axs[0].set_ylabel('Altitude (m)', fontsize=12)
    axs[0].set_xlabel('Counts/cm\u00b3')
    axs[0].set_title('N \u2265 10 nm')
    #axs[0].set_xlim(-50, 1500)
    
    ##--Second subplot: 2.5+ nm Particles vs Altitude--##
    axs[1].plot(binned_df['CPC3_conc_center'], binned_df['Altitude_center'], color='saddlebrown')
    axs[1].fill_betweenx(binned_df['Altitude_center'], binned_df['CPC3_conc_min'], 
                         binned_df['CPC3_conc_max'], color='sandybrown', alpha=0.25)
    axs[1].fill_betweenx(binned_df['Altitude_center'], binned_df['CPC3_conc_25th'],
                        binned_df['CPC3_conc_75th'], color='sandybrown', alpha=1)
    axs[1].set_title('N \u2265 2.5 nm')
    axs[1].set_xlabel('Counts/cm\u00b3')
    #axs[1].set_xlim(-50, 2000)
    
    ##--Third subplot: Nuc particles vs Altitude--##
    axs[2].plot(binned_df['nuc_particles_center'], binned_df['Altitude_center'], color='darkslategray')
    axs[2].fill_betweenx(binned_df['Altitude_center'], binned_df['nuc_particles_min'], 
                         binned_df['nuc_particles_max'], color='cadetblue', alpha=0.25)
    axs[2].fill_betweenx(binned_df['Altitude_center'], binned_df['nuc_particles_25th'],
                        binned_df['nuc_particles_75th'], color='cadetblue', alpha=1)
    
    ##--Plot uncertainty as its own trace--##
    axs[2].plot(binned_df['nuc_error_center'], binned_df['Altitude_center'], color='crimson', 
                linestyle='dashed', label='3$\sigma$ \nuncertainty')
    
    ##--Subscript 3-10--##
    axs[2].set_title('$N_{2.5-10}$')
    axs[2].set_xlabel('Counts/cm\u00b3')
    #axs[2].set_xlim(-50, 2000)
    
    ##--Fourth subplot: Aitken (grow) particles vs Altitude--##
    axs[3].plot(binned_averaged_df['grow_particles_center'], binned_averaged_df['Altitude_center'], color='darkslategray')
    axs[3].fill_betweenx(binned_averaged_df['Altitude_center'], binned_averaged_df['grow_particles_min'],
                          binned_averaged_df['grow_particles_max'], color='cadetblue', alpha=0.25)
    axs[3].fill_betweenx(binned_averaged_df['Altitude_center'], binned_averaged_df['grow_particles_25th'],
                         binned_averaged_df['grow_particles_75th'], color='cadetblue', alpha=1)
    
    ##--Plot uncertainty--##
    axs[3].plot(binned_averaged_df['grow_error_center'], binned_averaged_df['Altitude_center'], color='crimson',
                linestyle='dashed', label='3$\sigma$ \nuncertainty')
    
    ##--Subscript 10-130--##
    axs[3].set_title('$N_{10-130}$')
    axs[3].set_xlabel('Counts/cm\u00b3')
    
    axs[3].legend(loc='lower right')
    
    ##--Use f-string to embed flight # variable in plot title--##
    plt.suptitle(f"FIRE-ACE Vertical Particle Count Profile - {flight.replace('Flight', 'Flight ')} ({flight_date})", fontsize=16)
    
    ##--Adjusts layout to prevent overlapping--## 
    plt.tight_layout(rect=[0, -0.02, 1, 0.99])
    
    ##--Use f-string to save file with flight# appended--##
    #output_path = f"{output_path}\\{flight}"
    #plt.savefig(output_path, dpi=600, bbox_inches='tight') 
    
    plt.show()
    
    

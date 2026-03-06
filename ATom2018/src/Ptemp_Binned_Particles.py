# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 14:00:22 2026

@author: repooley
"""

import icartt
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
directory = r"C:\Users\repooley\REP_PhD\Arctic_NPF\ATom2018\data"

##--The flight10 file also includes flight11--##
flights_to_analyze = ["Flight2", "Flight10", "Flight11", "Flight12"]

##--Define function that creates datasets from filenames--##
def find_files(directory, flight):
    ##--flight data are stored in a folder called "raw"--##
    flight_dir = os.path.join(directory, "raw", flight)
    search_pattern = os.path.join(flight_dir, "*.ict")
    return sorted(glob.glob(search_pattern))

##--Loop through each flight in the list--##
for flight in flights_to_analyze:
    
    #########################
    ##--Open ICARTT Files--##
    #########################

    dataset = icartt.Dataset(find_files(directory, flight)[0])

    ####################################
    ##--Assign date to flight number--##
    ####################################
    
    if flight=="Flight2":
        flight_date = date(2018, 4, 27)
    elif flight=="Flight10":
        flight_date = date(2018, 4, 17)
    elif flight=="Flight11": 
        flight_date = date(2018, 5, 18)
    elif flight=="Flight12":
        flight_date = date(2018, 5, 19)
    
    #################
    ##--Pull data--##
    #################
    
    ##--AIMMS Data--##
    altitude = dataset.data['G_ALT'] # in m (not sure if this is best one)
    temperature = dataset.data['T'] # in K
    pressure = dataset.data['P'] * 100 # in Pa
    RH = dataset.data['Relative_Humidity'] # wrt water, percent
    time =dataset.data['UTC_Start'] # seconds since midnight UTC
    nucleating = dataset.data['N_nucl_AMP'] # num/cm^3 STP (2.7-12 nm)
    aitken = dataset.data['N_aitken_AMP'] # num/cm^3 STP (12-60 nm)
    
    ##--There are notable outliers in the nucleating data--##
    
    ##--First convert to a series for calc--##
    nucleating_series = pd.Series(nucleating)
    
    ##--REMOVE OUTLIERS above 99th percentile--##
    p = 0.99
    
    ##--Compute threshold for each UHSAS column--##
    nucleating_thresh = nucleating_series.quantile(p)
    
    ##--keep only rows where each bin is below its threshold--##
    nucleating_filtered = nucleating_series[nucleating_series.le(nucleating_thresh)]
    
    nucleating_filtered = nucleating_series.mask(
    nucleating_series <= nucleating_thresh)
    
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
    
    ##--Place in dataframe--##
    
    df = pd.DataFrame({'ptemp': potential_temp, 'nucleating': nucleating, 'aitken': aitken})
    
    #%%
    ###############
    ##--BINNING--##
    ###############
    
    ##--Define number of bins here--##
    num_bins = 124
    
    ##--Compute the minimum and maximum altitude, ignoring NaNs--##
    min_ptemp = df['ptemp'].min(skipna=True)
    max_ptemp = df['ptemp'].max(skipna=True)
    
    ##--Create bin edges from min_alt to max_alt--##
    bin_edges = np.linspace(min_ptemp, max_ptemp, num_bins + 1)
    
    ##--Pandas 'cut' splits altitude data into specified number of bins--##
    df['ptemp_bin'] = pd.cut(df['ptemp'], bins=bin_edges)
    
    ##--Group variables into each altitude bin--## 
    ##--Observed=false shows all bins, even empty ones--##
    binned_df = df.groupby('ptemp_bin', observed=False).agg(
        
       ##--Aggregate data by mean, min, and max--##
        ptemp_center=('ptemp', 'median'), 
        nuc_particles_center=('nucleating', 'median'), 
        nuc_particles_min=('nucleating', 'min'),
        nuc_particles_max=('nucleating', 'max'), 
        nuc_particles_25th=('nucleating', lambda x: x.quantile(0.25)),
        nuc_particles_75th=('nucleating', lambda x: x.quantile(0.75)),
        aitken_particles_center=('aitken', 'median'), 
        aitken_particles_min=('aitken', 'min'),
        aitken_particles_max=('aitken', 'max'), 
        aitken_particles_25th=('aitken', lambda x: x.quantile(0.25)),
        aitken_particles_75th=('aitken', lambda x: x.quantile(0.75))
    
        ##--Reset the index so Altitude_bin is just a column--##
    ).reset_index()
    
    #%%
    ################
    ##--PLOTTING--##
    ################
    
    ##--Creates figure with 4 horizontally stacked subplots sharing a y-axis--##
    fig, axs = plt.subplots(1, 2, figsize=(6, 6), sharey=True)
    
    ##--First subplot: 2.7-12 nm Particles vs Altitude--##
    
    ##--Averaged data in each bin is plotted against bin center--##
    axs[0].plot(binned_df['nuc_particles_center'], binned_df['ptemp_center'], color='maroon')
    ##--Range is given by filling between data minimum and maximum for each bin--##
    #axs[0].fill_betweenx(binned_df['Altitude_center'], binned_df['nuc_particles_min'], 
    #                     binned_df['nuc_particles_max'], color='indianred', alpha=0.25)
    axs[0].fill_betweenx(binned_df['ptemp_center'], binned_df['nuc_particles_25th'],
                        binned_df['nuc_particles_75th'], color='indianred', alpha=0.7)
    axs[0].set_ylabel('Potential Temperature (K)', fontsize=12)
    axs[0].set_xlabel('Counts/cm\u00b3')
    axs[0].set_title('$N_{2.7-12}$')
    #axs[0].set_xlim(-50, 1500)
    
    ##--Second subplot: 12-60 nm Paticles with Altitude--##
    axs[1].plot(binned_df['aitken_particles_center'], binned_df['ptemp_center'], color='maroon')
    ##--Range is given by filling between data minimum and maximum for each bin--##
    #axs[1].fill_betweenx(binned_df['Altitude_center'], binned_df['aitken_particles_min'], 
    #                     binned_df['aitken_particles_max'], color='indianred', alpha=0.25)
    axs[1].fill_betweenx(binned_df['ptemp_center'], binned_df['aitken_particles_25th'],
                        binned_df['aitken_particles_75th'], color='indianred', alpha=0.7)
    axs[1].set_xlabel('Counts/cm\u00b3')
    axs[1].set_title('$N_{12-60}$')
    #axs[0].set_xlim(-50, 1500)
    
    axs[1].legend(loc='lower right')
    
    ##--Use f-string to embed flight # variable in plot title--##
    plt.suptitle(f"ATom 4 Vertical Particle Count Profiles - {flight.replace('Flight', 'Flight ')} ({flight_date})", fontsize=16)
    
    ##--Adjusts layout to prevent overlapping--## 
    plt.tight_layout(rect=[0, -0.02, 1, 0.99])
    
    ##--Use f-string to save file with flight# appended--##
    #output_path = f"{output_path}\\CPC_Data_{flight}"
    #plt.savefig(output_path, dpi=600, bbox_inches='tight') 
    
    plt.show()
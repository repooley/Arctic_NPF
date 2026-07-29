# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 11:16:25 2025

@author: repooley
"""

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
directory = r"C:\Users\repooley\REP_PhD\Arctic_NPF\FIREACE1998\data"

##--Flights to analyze - flights 1-18--##
flights_to_analyze = ["Flight1", "Flight2", "Flight3",  
                      "Flight7", "Flight8", "Flight9", "Flight10", "Flight11", "Flight12",
                      "Flight13", "Flight14", "Flight15", "Flight16", "Flight17", "Flight18"]

##--Base output path in directory--##
#output_path = r"C:\Users\repooley\REP_PhD\Arctic_NPF\FIRACE1998\data\processed\VerticallyBinnedData"

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
    
    
    #############################
    ##--Propagate uncertainty--##        
    #############################
    
    ##--Use the 75th quartile median uncertainty from all of NETCARE--##
    nuc_error_3sigma = 133.71
    
    #######################################
    ##--Filter to NPF and non-NPF times--##
    #######################################
    
    ##--Temperature and PTemp--##
    
    temp_n_3_10 = pd.DataFrame({'Temp': temperature, 'PTemp': potential_temp, 'Nucleation': df['nuc_particles'],
                                     'LoD': nuc_error_3sigma})
    
       
    temp_npf = temp_n_3_10['Temp'][temp_n_3_10['Nucleation'] > temp_n_3_10['LoD']]
    ptemp_npf = temp_n_3_10['PTemp'][temp_n_3_10['Nucleation'] > temp_n_3_10['LoD']]
    
    temp_nonpf = temp_n_3_10['Temp'][temp_n_3_10['Nucleation'] <= temp_n_3_10['LoD']]
    ptemp_nonpf = temp_n_3_10['PTemp'][temp_n_3_10['Nucleation'] <= temp_n_3_10['LoD']]
    
    temp_df = {'NPF': temp_npf, 'No NPF': temp_nonpf}
    ptemp_df = {'NPF': ptemp_npf, 'No NPF': ptemp_nonpf}
       
    
    ##--Altitude--##
    
    alt_n_3_10 = pd.DataFrame({'Alt': altitude, 'Nucleation': df['nuc_particles'],
                                     'LoD': nuc_error_3sigma, 'PTemp': potential_temp})
    
            
    alt_npf = alt_n_3_10['Alt'][alt_n_3_10['Nucleation'] > alt_n_3_10['LoD']]
    alt_nonpf = alt_n_3_10['Alt'][alt_n_3_10['Nucleation'] <= alt_n_3_10['LoD']]
    alt_df = {'NPF': alt_npf, 'No NPF': alt_nonpf}
     
    
    ##--RH--##
    RH = pd.DataFrame({'RH': RH, 'Nucleation': df['nuc_particles'],
                                     'LoD': nuc_error_3sigma, 'PTemp': potential_temp})
    
        
    RH_npf = RH['RH'][RH['Nucleation'] > RH['LoD']]
    RH_nonpf = RH['RH'][RH['Nucleation'] <= RH['LoD']]
    RH_df = {'NPF': RH_npf, 'No NPF': RH_nonpf}
    
    
    ################
    ##--Plotting--##
    ################
    
    ##--Assign color palettes--## 
    palette = {'NPF':'#d92b3c', 'No NPF':'#931a25'}
    palette2 = {'NPF':'#c65e5e', 'No NPF':'#af3e3e'}
    palette3 = {'NPF':'#b11f84', 'No NPF':'#8c1868'}
    palette4 = {'NPF':'#fd5f5f', 'No NPF':'#fd2e2e'}
    palette5 = {'NPF':'#d17575', 'No NPF':'#c14545'}
    
    ##--TEMPERATURE--##
    
    fig, ax = plt.subplots(figsize = (4,6))
    ##--Cut=0 disallows interpolation beyond the data extremes. Remove inner box whiskers for clarity--##
    temp_plot = sns.violinplot(data=temp_df, palette=palette, ax=ax, cut=0, inner_kws={'whis_width': 0, 'solid_capstyle':'butt'})
    ax.set(xlabel='')
    ax.set(ylabel='Temperature (K)')
    
    ax.set(title=f"Temperature - {flight.replace('Flight', 'Flight ')} ({flight_date})")
        
    #plt.savefig(f"{output_path}\\temp/temp_{flight}", dpi=600)
    
    plt.show()
    
    fig, ax = plt.subplots(figsize=(4,6))
    ptemp_plot = sns.violinplot(data = ptemp_df, order=['NPF', 'No NPF'], palette=palette2,
                                ax=ax, cut=0, inner_kws={'whis_width': 0, 'solid_capstyle':'butt'})
    ax.set(xlabel='')
    ax.set(ylabel='Potential Temperature (K)')
    
     
    ax.set(title=f"Potential Temperature - {flight.replace('Flight', 'Flight ')} ({flight_date})")
        
    #plt.savefig(f"{output_path}\\ptemp/ptemp_{flight}", dpi=600)
    
    plt.show()
    
    ##--ALTITUDE--##
    
    fig, ax = plt.subplots(figsize=(4,6))
    alt_plot = sns.violinplot(data = alt_df, order=['NPF', 'No NPF'], palette=palette3,
                              ax=ax, cut=0, inner_kws={'whis_width': 0, 'solid_capstyle':'butt'})
    ax.set(xlabel='')
    ax.set(ylabel='Altitude AMSL (m)')
    
     
    ax.set(title=f"Altitude - {flight.replace('Flight', 'Flight ')} ({flight_date})")
    
    #plt.savefig(f"{output_path}\\altitude/alt_{flight}", dpi=600)
    
    plt.show()
    
    ##--RH--##
    
    fig, ax = plt.subplots(figsize=(4,6))
    alt_plot = sns.violinplot(data = RH_df, order=['NPF', 'No NPF'], palette=palette4,
                              ax=ax, cut=0, inner_kws={'whis_width': 0, 'solid_capstyle':'butt'})
    ax.set(xlabel='')
    ax.set(ylabel='RH with respect to water (%)')
    
    ax.set(title=f"RH w.r.t. Water - {flight.replace('Flight', 'Flight ')} ({flight_date})")
        
    #plt.savefig(f"{output_path}\\rh_water/rh_w_{flight}", dpi=600)
    
    plt.show()
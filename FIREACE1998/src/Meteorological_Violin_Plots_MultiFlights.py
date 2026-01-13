# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 13:09:30 2025

@author: repooley
"""

import icartt
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.stats import mannwhitneyu

###################
##--User inputs--##
###################

##--Set the base directory to project folder--##
directory = r"C:\Users\repooley\REP_PhD\Arctic_NPF\FIREACE1998\data"

##--Choose which flights to analyze here!--##
flights_to_analyze = ["Flight3", "Flight7", 
                      'Flight9', 'Flight10', 
                      'Flight11', 'Flight12', 'Flight13', 
                      'Flight14', 'Flight15', 'Flight16', 
                      'Flight17', 'Flight18']

##--Base output path for figures in directory--##
#output_path = r"C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data\processed\ViolinPlots\Meteorological"

################################
##--Open Files and Pull Data--##
################################

##--Define function that creates datasets from filenames--##
def find_files(directory, flight, partial_name):
    ##--flight data are stored in a folder called "raw"--##
    flight_dir = os.path.join(directory, "raw", flight)
    search_pattern = os.path.join(flight_dir, f"*{partial_name}*")
    return sorted(glob.glob(search_pattern))

###########################
##--Per-flight analysis--##
###########################

##--Store processed data for ALL flights here: --##
temps = []
ptemps = []
alts = []
RHs = []

##--Loop through each flight, pulling and analyzing data--##
for flight in flights_to_analyze:
    ##--Follow which flight is processing--##
    print(f"Processing {flight}...")
    
    
    ##--'raw' contains a 1hz and 2min datafile, the 1hz one is always first--##
    path = find_files(directory, flight, "FIREACE")[0]
    data = pd.read_csv(path)

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
    df = pd.DataFrame({'Alt': altitude, 'PTemp': potential_temp, 
                       'Temp': temperature, 'RH': RH, 'CPC3_conc':CPC3_conc_STP, 'CPC10_conc': CPC10_conc_STP})

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
    
    ##--Add error to dataframe--##
    df['LoD'] = nuc_error_3sigma
    
    ##--Create a boolean mask for NPF based on LoD--##
    df['NPF'] = df['nuc_particles'] > df['LoD']
    
    ##--Rename the boolean mask NPF or No NPF--##
    df['NPF'] = df['NPF'].map({True: 'NPF', False: 'No NPF'})
    
    #######################################
    ##--Filter to NPF and non-NPF times--##
    #######################################

    temp_df = df[['Temp', 'PTemp', 'nuc_particles', 'LoD', 'NPF']]
    ptemp_df = df[['PTemp', 'nuc_particles', 'LoD', 'NPF']]
    alt_df = df[['Alt', 'PTemp', 'nuc_particles', 'LoD', 'NPF']]
    RH_df = df[['RH', 'PTemp', 'nuc_particles', 'LoD', 'NPF']]

    
    ##--Append to correct regional list--##
    
    temps.append(temp_df)
    ptemps.append(ptemp_df)
    RHs.append(RH_df)
    alts.append(alt_df)

#######################################
##--Filter to NPF and non-NPF times--##
#######################################

##--Concatenate all flight data--##
temps = pd.concat(temps, ignore_index=True)
ptemps = pd.concat(ptemps, ignore_index=True)
alts = pd.concat(alts, ignore_index=True)
RHs = pd.concat(RHs, ignore_index=True)

##--Filter into NPF and non-NPF subsets--##
temp_npf = temps.loc[temps['NPF'] == 'NPF', 'Temp']
temp_nonpf = temps.loc[temps['NPF'] == 'No NPF', 'Temp']

ptemp_npf = ptemps.loc[ptemps['NPF'] == 'NPF', 'PTemp']
ptemp_nonpf = ptemps.loc[ptemps['NPF'] == 'No NPF', 'PTemp']

alt_npf = alts.loc[alts['NPF'] == 'NPF', 'Alt']
alt_nonpf = alts.loc[alts['NPF'] == 'No NPF', 'Alt']

RH_npf = RHs.loc[RHs['NPF'] == 'NPF', 'RH']
RH_nonpf = RHs.loc[RHs['NPF'] == 'No NPF', 'RH']

##--Counts--##
temp_npf_count = len(temp_npf)
temp_nonpf_count = len(temp_nonpf)

ptemp_npf_count = len(ptemp_npf)
ptemp_nonpf_count = len(ptemp_nonpf)

alt_npf_count = len(alt_npf)
alt_nonpf_count = len(alt_nonpf)

RH_npf_count = len(RH_npf)
RH_nonpf_count = len(RH_nonpf)

################
##--Plotting--##
################

##--Order of label appearances:--##
##--NPF column stores boolean true or false--##
group_order = ['NPF', 'No NPF']

##--Palette for temperature plot--##
palette = {'NPF':'#d92b3c', 'No NPF': '#931A25'}

fig, ax = plt.subplots(figsize = (6,8))
##--Cut=0 disallows interpolation beyond the data extremes--##
temp_plot = sns.violinplot(data=temps, x='NPF', y='Temp', hue='NPF', order = group_order, legend=False,
                           inner_kws={'whis_width': 0, 'solid_capstyle':'butt'}, palette=palette, ax=ax, cut=0)

plt.text(0.26, 0.125, "N={}".format(temp_npf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')
plt.text(0.64, 0.125, "N={}".format(temp_nonpf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')

ax.set_xlabel('')
ax.set_xticks(range(len(group_order)))
plt.tick_params(labelsize=14)
ax.set_xticklabels(group_order)

plt.ylabel('Temperature (K)', fontsize=14)
plt.title('FIRE-ACE Temperature', fontsize=14)
 
#plt.savefig(f"{output_path}\\temp/temp_MultiFlights", dpi=600)

plt.show()

##--Complementary palette for ptemp--##
palette2 = {'NPF': '#c65e5e', 'No NPF':'#af3e3e'}

fig, ax = plt.subplots(figsize=(6,8))
ptemp_plot = sns.violinplot(data = ptemps, x='NPF', y='PTemp', hue='NPF', order=group_order, legend=False,
                                  inner_kws={'whis_width': 0, 'solid_capstyle':'butt'}, palette=palette2, ax=ax, cut=0)

plt.text(0.26, 0.125, "N={}".format(ptemp_npf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')
plt.text(0.64, 0.125, "N={}".format(ptemp_nonpf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')

ax.set(xlabel='')
ax.set_xticks(range(len(group_order)))
plt.tick_params(labelsize=14)
ax.set_xticklabels(group_order)

plt.ylabel('Potential Temperature (K)', fontsize=14)


plt.title('FIRE-ACE Potential Temperature', fontsize=14)


#plt.savefig(f"{output_path}\\ptemp/ptemp_MultiFlights", dpi=600)

plt.show()

##--Palette for altitude--##
palette3 = {'NPF': '#b11f84', 'No NPF':'#8c1868'}

fig, ax = plt.subplots(figsize=(6,8))
alt_plot = sns.violinplot(data = alts, x='NPF', y='Alt', hue='NPF', order=group_order, legend=False,
                                  inner_kws={'whis_width': 0, 'solid_capstyle':'butt'}, palette=palette3, ax=ax, cut=0)

plt.text(0.26, 0.125, "N={}".format(alt_npf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')
plt.text(0.64, 0.125, "N={}".format(alt_nonpf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')


ax.set(xlabel='')
ax.set_xticks(range(len(group_order)))
plt.tick_params(labelsize=14)
ax.set_xticklabels(group_order)

plt.ylabel('Altitude M.A.S.L. (m)', fontsize=14)

plt.title('FIRE-ACE Altitude', fontsize=14)
  
#plt.savefig(f"{output_path}\\altitude/alt_MultiFlights", dpi=600)

plt.show()

##--Palette for RH--##
palette4 = {'NPF': '#fd5f5f', 'No NPF':'#fd2e2e'}

fig, ax = plt.subplots(figsize=(6,8))
RH_plot = sns.violinplot(data = RHs, x='NPF', y='RH', hue='NPF', order=group_order, legend=False,
                                  inner_kws={'whis_width': 0, 'solid_capstyle':'butt'}, palette=palette4, ax=ax, cut=0)

plt.text(0.26, 0.125, "N={}".format(RH_npf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')
plt.text(0.64, 0.125, "N={}".format(RH_nonpf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')

ax.set(xlabel='')
ax.set_xticks(range(len(group_order)))
plt.tick_params(labelsize=14)
ax.set_xticklabels(group_order)

plt.ylabel('Relative Humidity (%)', fontsize=14)

plt.title("FIRE-ACE Relative Humidity w.r.t. Water", fontsize=14)
   
#plt.savefig(f"{output_path}\\rh_water/rh_w_MultiFlights", dpi=600)

plt.show()

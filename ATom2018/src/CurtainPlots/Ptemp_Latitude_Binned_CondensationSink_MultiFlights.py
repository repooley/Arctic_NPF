# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 08:57:30 2026

@author: repooley
"""

import icartt
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from scipy.stats import binned_statistic_2d
from datetime import date
import matplotlib.ticker as ticker
import cmcrameri as cm # pretty colors

###################
##--User inputs--##
###################

##--Set the base directory to project folder--##
directory = r"C:\Users\repooley\REP_PhD\Arctic_NPF\ATom2018\data"

##--The flight10 file also includes flight11--##
flights_to_analyze = ["Flight2", "Flight10", "Flight11", "Flight12"]

##--Set binning for PTemp and Latitude--##
num_bins_lat = 6
num_bins_ptemp = 6

##--Define function that creates datasets from filenames--##
def find_files(directory, flight):
    ##--flight data are stored in a folder called "raw"--##
    flight_dir = os.path.join(directory, "raw", flight)
    search_pattern = os.path.join(flight_dir, "*.ict")
    return sorted(glob.glob(search_pattern))

all_diameter_dfs = []
condensation_sinks = []

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
    latitude = dataset.data['G_LAT'] # deg
    temperature = dataset.data['T'] # in K
    pressure = dataset.data['P'] * 100 # in Pa
    RH = dataset.data['Relative_Humidity'] # wrt water, percent
    time =dataset.data['UTC_Start'] # seconds since midnight UTC
    nucleating = dataset.data['N_nucl_AMP'] # num/cm^3 STP (2.7-12 nm)
    aitken = dataset.data['N_aitken_AMP'] # num/cm^3 STP (12-60 nm)
    accumulation = dataset.data['N_accum_AMP'] # num/cm^3 STP (60 nm - 0.5 um)
    coarse = dataset.data['N_coarse_AMP'] # num/cm^3 STP (0.5 um - 4.8 um)
    
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
        
    ###########################
    ##--Wrangle binned data--##
    ###########################
    
    ##--Concatenate bin edges--##
    combined_bin_edges = np.concatenate([
        [12],       # upper edge of N(2.7-12), also lower of next
        [60],       # upper edge of N(12-60), also lower of next
        [500],      # upper edge of N(60-500), also lower of next
        [4800]      # upper edge of final bin
        ])
    
    ##--Concatenate bin centers--##
    bin_centers = np.concatenate([
        [36], 
        [280],
        [2650]
        ])
    
    ##--Place all binned data in a single df--##
    all_bins_aligned = pd.concat([pd.DataFrame({'36':aitken, '280':accumulation, '2650':coarse}, index=time)], axis=1)
    total_particle_count = all_bins_aligned.sum(axis=1, numeric_only=True) 
    
    ##--Create a dictionary to store each column as a separate dataframe, col names are keys--##
    diameter_dfs = {col: pd.DataFrame({col: all_bins_aligned[col]}) for col in all_bins_aligned.columns}
    
    all_diameter_dfs.append(diameter_dfs)
    
    
    ######################################
    ##--Condensation sink calculations--##
    ######################################
    
    ##--Constants--##
    
    R = 8.314 # Ideal gas constant (m^3*Pa*K^-1*mol^-1)
    ##--H2SO4 kinetic diam: lifted from Williamson et al for now (avg of their values)--##
    Ds = 5.49E-10 # in m
    ##--Kinetic diam of air calculated from mixing ratios and dataset on Wikepedia--##
    Dair = 3.61E-10 # in m
    avg_diam = (Ds + Dair)/2
    ##--Mass sulfuric acid--##
    Ms = 98.079 # g/mol
    ##--Mass air--##
    Mair = 28.96 # g/mol
    ##--Reduced mass--##
    Z = Ms/Mair 
    ##--Sticking coefficient - fair to assume unity for H2SO4--##
    alpha = 1
    ##--Boltzmann--##
    k = 1.38E-23 # J/K
    ##--Sutherland's law for dynamic viscosity--##
    C = 1.458E-6 # kg/ms*sqrt(K)
    S = 110.4 # K
    
    ##--Variables--##
    
    ##--Convert temperature and pressure from numpy array to dataframe to subvert errors--##
    temperature_series = pd.Series(temperature, index=time)
    pressure_series = pd.Series(pressure, index=time)
    
    ##--Loop through dfs in diameter_dfs and calculate needed variables for each bin--##
    ##--Store in series initialized at zero--##
    condensation_sink = pd.Series(0, index=time)
    
    for diameter, df in diameter_dfs.items():
        
        ##--Convert column diams from string to float--##
        mean_diameter = (float(diameter)) * 1E-9 # in m
        
        ##--Calculate mean free path of H2SO4 from molecular diameter--##
        df['mean_free_path'] = ((R * temperature_series.loc[df.index]) / ((2 ** (1/2)) 
                                * 3.14159 * (Ds ** 2) * 6.022E23 * pressure_series.loc[df.index])) # in m/molecule
        
        ##--Calculate the Knudsen number--##
        df['Knudsen_num'] = df['mean_free_path'] / (mean_diameter / 2) # unitless ratio
        
        ##--Calculate Fuch's correction--##
        df['Fuchs_correction'] = (1 + df['Knudsen_num']) / (1 + ((4/(3*alpha)) 
                                + 0.337) * df['Knudsen_num'] + (4/(3*alpha) * (df['Knudsen_num']) ** 2)) # unitless
        
        ##--Calculate slip correction for Dynamic Viscosity calculation--##
        df['Slip_correction'] = (1 + df['Knudsen_num'] * (2.514 + 0.800 * (np.exp(-0.550 / df['Knudsen_num'])))) # unitless
        
        ##--Calculate dynamic viscosity using Sutherland's law with constants for air--##
        df['Dynamic_viscosity'] = (C * (temperature_series.loc[df.index]) ** (3/2)) / (temperature_series.loc[df.index] + S)
        
        ##--Calculate the diffusion coefficient--##
        df['Diffusion_coefficient'] = ((k * temperature_series.loc[df.index] * df['Slip_correction']) / 
                                       (3 * 3.14159 * df['Dynamic_viscosity'] * Ds)) # m^2/s
        
        ##--Extract Particle Concentration (first column in diameter_dfs)--##
        df['Particle_concentration'] = df.iloc[:, 0] / 1E-6 # converted to #/m^3
        
        ##--Per-bin contribution to condensation sink (before final multiplication)--##
        df['CS_contribution'] = (df['Fuchs_correction'] * mean_diameter * df['Particle_concentration'])
        
        ##--Multiply each bin’s CS contribution by its diffusion coefficient--##
        ##--Fill NaN values in CS_contribution with zeros to prevent NaN result--##
        condensation_sink += (2 * np.pi * df['Diffusion_coefficient'] * df['CS_contribution']).fillna(0)
     
    ##--Populate series--##
    condensation_sink = pd.DataFrame({'Condensation_Sink': condensation_sink}) 
    
    ##--Append latitude data--##
    condensation_sink['Latitude'] = latitude
    
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
        
    condensation_sink['PTemp'] = potential_temp
    
    ##--Append condensation_sinks with the columns--##
    condensation_sinks.append(condensation_sink.dropna(subset=['Condensation_Sink', 'Latitude', 'PTemp']))


###########################
##--Create 2D histogram--##
###########################

##--Float type NaNs in potential_temp cannot convert to int, so must be removed--##
all_latitudes_CS10 = np.concatenate([df['Latitude'].values for df in condensation_sinks])
all_ptemps_CS10 = np.concatenate([df['PTemp'].values for df in condensation_sinks])
all_CS10 = np.concatenate([df['Condensation_Sink'].values for df in condensation_sinks])
 
lat_bin_edges_CS10 = np.linspace(all_latitudes_CS10.min(), all_latitudes_CS10.max(), num_bins_lat + 1)
ptemp_bin_edges_CS10 = np.linspace(all_ptemps_CS10.min(), all_ptemps_CS10.max(), num_bins_ptemp + 1)
 
CS10_bin_medians, _, _, _ = binned_statistic_2d(all_latitudes_CS10, all_ptemps_CS10, 
        all_CS10, statistic="median", bins=[lat_bin_edges_CS10, ptemp_bin_edges_CS10])

################
##--PLOTTING--##
################

##--Particles larger than 3 nm--##
fig1, ax1 = plt.subplots(figsize=(6, 6))

##--Make special color map where 0 values are white--##
new_cmap = cm.cm.batlow
##--Values under specified minimum will be white--##
new_cmap.set_under('w')

##--Use pcolormesh for the plot, set minimum value for viridis colors as 1--##
CS10_plot = ax1.pcolormesh(lat_bin_edges_CS10, ptemp_bin_edges_CS10, CS10_bin_medians.T,  # Transpose to align correctly
    shading='auto', cmap=new_cmap)#, vmin=0, vmax=0.006)


##--Add colorbar--##
cb = fig1.colorbar(CS10_plot, ax=ax1, orientation='horizontal', location='bottom', pad=0.15)
cb.minorticks_on()
cb.ax.tick_params(labelsize=18)
cb.set_label('Condensation $s^{-1}$', fontsize=18)

##--Set axis labels--##
ax1.set_xlabel('Latitude (°)', fontsize=18)
ax1.set_ylabel('Potential Temperature \u0398 (K)', fontsize=18)
ax1.tick_params(axis='both', labelsize=18)
ax1.set_title("Condensation Sink for $N_{2.5-10}$", fontsize=20)
#ax1.set_ylim(238, 316)
#ax1.set_xlim(64, 86)
ax1.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax1.yaxis.set_major_locator(ticker.MultipleLocator(10))

##--Use f-string to save file with flight# appended--##
#CS10_output_path = f"{output_path}\\{flight}_MultiFlights"
#plt.savefig(CS10_output_path, dpi=600, bbox_inches='tight') 

plt.tight_layout()
plt.show()

########################
##--Diagnostic Plots--##
########################

##--Remove hashtags below to comment out this section--##
#'''

##--Counts per bin for CPC3 data--##
CS10_bin_counts, _, _, _ = binned_statistic_2d(all_latitudes_CS10, 
    all_ptemps_CS10, all_CS10, statistic='count', bins=[lat_bin_edges_CS10, ptemp_bin_edges_CS10])

##--Particles larger than 3 nm--##
fig1, ax1 = plt.subplots(figsize=(8, 6))

##--Make special color map where 0 values are white--##
new_cmap = plt.get_cmap('inferno')
##--Values under specified minimum will be white--##
new_cmap.set_under('w')

##--Use pcolormesh for the plot, set minimum value for viridis colors as 1--##
CS10_plot = ax1.pcolormesh(lat_bin_edges_CS10, ptemp_bin_edges_CS10, CS10_bin_counts.T,  # Transpose to align correctly
    shading='auto', cmap=new_cmap)#, vmin=1, vmax=10000)

##--Add colorbar--##
cb = fig1.colorbar(CS10_plot, ax=ax1)
cb.minorticks_on()
cb.ax.tick_params(labelsize=16)
cb.set_label('Number of Data Points', fontsize=16)

# Set axis labels
ax1.set_xlabel('Latitude (°)', fontsize=16)
ax1.set_ylabel('Potential Temperature \u0398 (K)', fontsize=16)
ax1.tick_params(axis='both', labelsize=16)
ax1.set_title("Condensation Sink Counts per Bin", fontsize=18)
#ax1.set_ylim(238, 301)
#ax1.set_xlim(79.5, 83.7)

##--Use f-string to save file with flight# appended--##
#CS10_diag_output_path = f"{output_path}\\MultiFlights_diagnostic"
#plt.savefig(CS10_diag_output_path, dpi=600, bbox_inches='tight') 

plt.tight_layout()
plt.show()
#'''
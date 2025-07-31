# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 11:55:14 2025

@author: repooley
"""

##--This script also includes rBC--##

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
directory = r"C:\Users\repooley\REP_PhD\NETCARE2015\data"

##--Select flight (Flight2 thru Flight10)--##
##--FLIGHT1 HAS NO UHSAS FILES--##
flight = "Flight2"

##--Base output path for figures in directory--##
output_path = r"C:\Users\repooley\REP_PhD\NETCARE2015\data\processed\ViolinPlots"

#########################
##--Open ICARTT Files--##
#########################

##--Define function that creates datasets from filenames--##
def find_files(directory, flight, partial_name):
    ##--flight data are stored in a folder called "raw"--##
    flight_dir = os.path.join(directory, "raw", flight)
    search_pattern = os.path.join(flight_dir, f"*{partial_name}*")
    return sorted(glob.glob(search_pattern))

##--Meterological data from AIMMS monitoring system--##
aimms = icartt.Dataset(find_files(directory, flight, "AIMMS_POLAR6")[0])

##--Black carbon data from SP2--##
SP2 = icartt.Dataset(find_files(directory, flight, "SP2_Polar6")[0])

##--UHSAS data--##
UHSAS = icartt.Dataset(find_files(directory, flight, 'UHSAS')[0])

##--OPC data--##
OPC = icartt.Dataset(find_files(directory, flight, 'OPC')[0])

##--CPC data--##
CPC10 = icartt.Dataset(find_files(directory, flight, 'CPC3772')[0])
CPC3 = icartt.Dataset(find_files(directory, flight, 'CPC3776')[0])

#########################
##--Pull & align data--##
#########################

##--AIMMS Data--##
altitude = aimms.data['Alt'] # in m
latitude = aimms.data['Lat'] # in degrees
temperature = aimms.data['Temp'] + 273.15 # in K
pressure = aimms.data['BP'] # in pa
aimms_time =aimms.data['TimeWave'] # seconds since midnight

##--Establish AIMMS start/stop times--##
aimms_end = aimms_time.max()
aimms_start = aimms_time.min()

##--Black carbon--##
BC_count = SP2.data['BC_numb_concSTP'] # in STP

##--Handle black carbon data with different start/stop times than AIMMS--##
BC_time = SP2.data['Time_UTC']

##--Trim CO data if it starts before AIMMS--##
if BC_time.min() < aimms_start:
    mask_start = BC_time >= aimms_start
    BC_time = BC_time[mask_start]
    BC_count = BC_count[mask_start]
    
##--Append CO data with NaNs if it ends before AIMMS--##
if BC_time.max() < aimms_end: 
    missing_times = np.arange(BC_time.max()+1, aimms_end +1)
    BC_time = np.concatenate([BC_time, missing_times])
    BC_count = np.concatenate([BC_count, [np.nan]*len(missing_times)])

##--Create a DataFrame for BC data and reindex to AIMMS time, setting non-overlapping times to nan--##
BC_df = pd.DataFrame({'Time_UTC': BC_time, 'BC_count': BC_count})
BC_aligned = BC_df.set_index('Time_UTC').reindex(aimms_time)
BC_aligned['BC_count']= BC_aligned['BC_count'].where(BC_aligned.index.isin(aimms_time), np.nan)
BC_count_aligned = BC_aligned['BC_count']

##--USHAS Data--##
UHSAS_time = UHSAS.data['time'] # seconds since midnight

##--Bin data are in a CSV file--##
UHSAS_bins = pd.read_csv(r"C:\Users\repooley\REP_PhD\NETCARE2015\data\raw\NETCARE2015_UHSAS_bins.csv")

##--Make list of columns to pull, each named bin_x--##
##--Bins 1-13 not trustworthy. Bins 76-99 overlap with OPC, discard--##
##--Trim to use bins 14-76 (500>85 nm)--##
UHSAS_bin_num = [f'bin_{i}' for i in range(14, 75)]

##--Information for bins 14 thru 99--##
UHSAS_bin_center = UHSAS_bins['bin_avg'].iloc[14:75]
UHSAS_lower_bound = UHSAS_bins['lower_bound'].iloc[14:75]
UHSAS_upper_bound = UHSAS_bins['upper_bound'].iloc[14:75]

##--Put column names and content in a dictionary and then convert to a Pandas df--##
UHSAS_bins = pd.DataFrame({col: UHSAS.data[col] for col in UHSAS_bin_num})

##--Create new column names by rounding the bin center values to the nearest integer--##
UHSAS_new_col_names = UHSAS_bin_center.round().astype(int).tolist()

##--Rename the UHSAS_bins df columns to bin average values--##
UHSAS_bins.columns = UHSAS_new_col_names

##--Add time, total_num to UHSAS_bins df--##
UHSAS_bins.insert(0, 'Time', UHSAS_time)

##--Align UHSAS_bins time to AIMMS time--##
UHSAS_bins_aligned = UHSAS_bins.set_index('Time').reindex(aimms_time)

##--Tabulate total count across all bins--##
UHSAS_total_num = UHSAS_bins_aligned.sum(axis=1, numeric_only=True)

##--OPC Data--##
OPC_time = OPC.data['Time_UTC'] # seconds since midnight

##--Bin data are in a CSV file--##
OPC_bin_info = pd.read_csv(r"C:\Users\repooley\REP_PhD\NETCARE2015\data\raw\NETCARE2015_OPC_bins.csv")

##--Select bins greater than 500 nm (Channel 7 and greater)--##
OPC_bin_center = OPC_bin_info['bin_avg'].iloc[6:31]
OPC_lower_bound = OPC_bin_info['lower_bound'].iloc[6:31]
OPC_upper_bound = OPC_bin_info['upper_bound'].iloc[6:31]

##--Make list of columns to pull, each named Channel_x--##
OPC_bin_num = [f'Channel_{i}' for i in range(7, 32)]

##--Put column names and content in a dictionary and then convert to a Pandas df--##
OPC_bins = pd.DataFrame({col: OPC.data[col] for col in OPC_bin_num})

##--Create new column names by rounding the bin center values to the nearest integer--##
OPC_new_col_names = OPC_bin_center.round().astype(int).tolist()

##--Rename the OPC_bins df columns to bin average values--##
OPC_bins.columns = OPC_new_col_names

##--Add time, total_num to OPC_bins df--##
OPC_bins.insert(0, 'Time', OPC_time)

##--Align OPC_bins time to AIMMS time--##
OPC_bins_aligned = OPC_bins.set_index('Time').reindex(aimms_time)

##--OPC samples every six seconds. Most rows are NaN--##
##--Forward-fill NaN values to propagate last valid reading--##
##--Limit forward filling to 5 NaN rows--##
OPC_bins_filled = OPC_bins_aligned.ffill(limit=5)

##--10 nm CPC data--##
CPC10_time = CPC10.data['time']
CPC10_conc = CPC10.data['conc'] # count/cm^3

##--2.5 nm CPC data--##
CPC3_time = CPC3.data['time']
CPC3_conc = CPC3.data['conc'] # count/cm^3

##--Make CPC3 df and set index to CPC3 time--##
CPC3_df = pd.DataFrame({'time': CPC3_time, 'conc': CPC3_conc}).set_index('time')
##--Make a new df reindexed to aimms_time. Populate with CPC3 conc--##
CPC3_conc_aligned = CPC3_df.reindex(aimms_time)['conc']

##--Make CPC10 df and set index to CPC10 time--##
CPC10_df = pd.DataFrame({'time': CPC10_time, 'conc': CPC10_conc}).set_index('time')
##--Make a new df reindexed to aimms_time. Populate with CPC10 conc--##
CPC10_conc_aligned = CPC10_df.reindex(aimms_time)['conc']

###############################
##--De-Normalize UHSAS Data--##
###############################

##--Calculate dlogDp for UHSAS bins--##
UHSAS_dlogDp = np.log(UHSAS_upper_bound.values) - np.log(UHSAS_lower_bound.values)

##--Get only particle count data (excluding 'Time')--##
UHSAS_particle_counts = UHSAS_bins_aligned.loc[:, UHSAS_new_col_names]  # Adjust column names as needed

##--De-Normalize counts by multiplying by dlogDp across all rows--##
UHSAS_denorm_counts = UHSAS_particle_counts.multiply(UHSAS_dlogDp, axis=1)

##--Take out of STP--##
P_STP = 101325  # Pa
T_STP = 273.15  # K

##--Create empty list for OPC particles--##
UHSAS_abs_counts = []

for UHSAS, T, P in zip(UHSAS_denorm_counts.values, temperature, pressure):
    if np.isnan(T) or np.isnan(P):
        ##--Append with NaN if any input is NaN--##
        UHSAS_abs_counts.append([np.nan]*len(UHSAS))
    else:
        ##--Perform conversion if all inputs are valid--##
        corrected_UHSAS = UHSAS / (P_STP / P) / (T / T_STP)
        UHSAS_abs_counts.append(corrected_UHSAS)

##--Convert back to DataFrame with same columns and index--##
UHSAS_abs_counts = pd.DataFrame(UHSAS_abs_counts, columns=UHSAS_denorm_counts.columns, index=UHSAS_denorm_counts.index)

##--Reset index of UHSAS_abs_counts to align with time--##
min_length = min(len(UHSAS_time), len(UHSAS_abs_counts))
UHSAS_time = UHSAS_time[:min_length]
UHSAS_abs_counts = UHSAS_abs_counts.iloc[:min_length]

######################
##--Calc N(2.5-10)--##
######################

##--Convert to STP--##
##--Create empty list for CPC3 particles--##
CPC3_conc_STP = []

for CPC3, T, P in zip(CPC3_conc_aligned, temperature, pressure):
    if np.isnan(CPC3) or np.isnan(T) or np.isnan(P):
        ##--Append with NaN if any input is NaN--##
        CPC3_conc_STP.append(np.nan)
    else:
        ##--Perform conversion if all inputs are valid--##
        CPC3_conversion = CPC3 * (P_STP / P) * (T / T_STP)
        CPC3_conc_STP.append(CPC3_conversion)
    
##--Create empty list for CPC10 particles--##
CPC10_conc_STP = []

for CPC10, T, P in zip(CPC10_conc_aligned, temperature, pressure):
    if np.isnan(CPC10) or np.isnan(T) or np.isnan(P):
        ##--Append with NaN if any input is NaN--##
        CPC10_conc_STP.append(np.nan)
    else:
        ##--Perform conversion if all inputs are valid--##
        CPC10_conversion = CPC10 * (P_STP / P) * (T / T_STP)
        CPC10_conc_STP.append(CPC10_conversion)

##--Creates a Pandas dataframe for CPC data--##
CPC_df = pd.DataFrame({'Altitude': altitude, 'Latitude': latitude, 'CPC3_conc':CPC3_conc_STP, 'CPC10_conc': CPC10_conc_STP})

##--Calculate N3-10 particles--##
nuc_particles = (CPC_df['CPC3_conc'] - CPC_df['CPC10_conc'])

##--Change calculated particle counts less than zero to NaN--##
nuc_particles = np.where(nuc_particles >= 0, nuc_particles, np.nan)

##--Put N(2.5-10) bin center in a df--##
n_3_10_center = pd.DataFrame([6.25]) # Approximate mean of 2.5 and 10

##--Create a dataframe for N 2.5-10--##
n_3_10 = pd.DataFrame({'time': aimms_time, '6': nuc_particles}).set_index('time')

#####################
##--Calc N(10-89)--##
#####################

##--Create df with UHSAS total counts--##
UHSAS_total = pd.DataFrame({'Time': UHSAS_time, 'Total_count': UHSAS_abs_counts.sum(axis=1)})

##--Reindex UHSAS_total df to AIMMS time--##
UHSAS_total_aligned = UHSAS_total.set_index('Time').reindex(aimms_time)

##--Create df with CPC10 counts and set index to time--##
CPC10_counts = pd.DataFrame({'Time':aimms_time, 'Counts':CPC10_conc_aligned}).set_index('Time')

##--Calculate particles below UHSAS lower cutoff--##
n_10_89 = (CPC10_counts['Counts'] - UHSAS_total_aligned['Total_count'])

##--Change calculated particle counts less than zero to NaN--##
n_10_89 = np.where(n_10_89 >= 0, n_10_89, np.nan)

##--Put N(10-89) bin center in a df--##
n_10_89_center = pd.DataFrame([49.5])

##--Convert n_10_89 to a df--##
n_10_89 = pd.DataFrame({'49.5': n_10_89, 'time':aimms_time}).set_index('time')

##--Change first column name from string to integer--##
n_10_89.columns = [49.5]

###########################
##--Wrangle binned data--##
###########################

##--Concatenate bin edges--##
combined_bin_edges = np.concatenate([
    [2.5],      # start of first bin
    [10],       # upper edge of N(2.5-10), also lower of next
    [89.32],       # upper edge of N(10-89), also lower of next
    UHSAS_upper_bound.values,  # UHSAS bins continue from 85
    OPC_upper_bound.values     # OPC bins continue from last UHSAS
])

##--Calculate time edges for each bin--##
time_step = aimms_time[1] - aimms_time[0]  
time_edges = np.append(aimms_time, aimms_time[-1] + time_step)  # length N + 1

##--Concatenate bin centers and reindex--##
bin_centers = pd.concat([n_10_89_center, UHSAS_bin_center, OPC_bin_center], axis=0).reset_index(drop=True)

##--Place all binned data in a single df--##
all_bins_aligned = pd.concat([n_10_89, UHSAS_bins_aligned, OPC_bins_filled], axis=1)
total_particle_count = all_bins_aligned.sum(axis=1, numeric_only=True) 

##--Create a dictionary to store each column as a separate dataframe, col names are keys--##
diameter_dfs = {col: pd.DataFrame({col: all_bins_aligned[col]}) for col in all_bins_aligned.columns}


######################################
##--Condensation sink calculations--##
######################################

##--USING ABSOLUTE COUNT DATA--##

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
temperature_series = pd.Series(temperature, index=aimms_time)
pressure_series = pd.Series(pressure, index=aimms_time)

##--Loop through dfs in diameter_dfs and calculate needed variables for each bin--##
##--Store in series initialized at zero--##
condensation_sink = pd.Series(0, index=aimms_time)  

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

#####################################
##--Coagulation sink calculations--##
#####################################

##--USING ABSOLUTE COUNT DATA--##

##--Constants--##

##--Ideal gas--##
R = 8.314 # m^3*Pa*K^-1*mol^-1
##--Boltzmann--##
k = 1.38E-23 # m^2*kg*s^-2*K^-1
##--Sutherland's law for dynamic viscosity--##
C = 1.458E-6 # kg/ms*sqrt(K)
S = 110.4 # K
##--Molar mass air molecules--##
MMair = 28.96 # g/mol
##--Mass single air molecule in kg--##
Mair = MMair/(6.022E23 * 1000) # kg
##--Diameter average air molecule--##
Dair = 3.61E-10 # in m

##--For N(2.5-10)--##

##--Median particle diameter--##
nuc_diam = 6.25E-9 # m

##--Vol spherical particle--##
nuc_vol = (4/3) * np.pi * (nuc_diam / 2) ** 3 #m^3

##--Mass particle assuming density of 1--##
nuc_mass = nuc_vol #kg

##--Reduced mass ratio--##
z_nuc = nuc_mass/Mair

##--Collision cross section nuc particle + air--##
sigma_nuc = (Dair + nuc_diam) / 2

##--Variables--##

##--Convert temperature and pressure from numpy array to dataframe to subvert errors--##
temperature_series = pd.Series(temperature, index=aimms_time)
pressure_series = pd.Series(pressure, index=aimms_time)
##--Concentration air molecules--##
Nair = (6.022E23 * pressure) / (R * temperature) # num/m^3

##--For N(2.5-10)--##

##--Mean particle speed--##
nuc_speed = ((8 * k * temperature_series) / (np.pi * nuc_mass))**(1/2)

##--Collision cross section, assumes collision with equal size particle--##
nuc_collision_cross = np.pi * (nuc_diam )**2 # m^2

##--Estimate of mean free path against air for use in slip correction--##
nuc_mfp_estimate = 1/(np.pi * (1 + z_nuc)**(1/2) * Nair * sigma_nuc**2) # m

##--Knudsen number--##
nuc_knudsen = nuc_mfp_estimate / (nuc_diam/2) # unitless 

##--Cunningham slip correction--##
nuc_slip = 1 + 2 * nuc_knudsen * (2.514 + 0.800 * np.exp(-0.550 / nuc_knudsen))

##--Dynamic viscosity--##
dynam_viscosity = (C * temperature_series ** (3/2)) / (temperature_series + S)

##--Diffusivity--##
nuc_diffusivity = (k * temperature_series * nuc_slip / (3 * np.pi * dynam_viscosity * nuc_diam)) # m^2/s

##--Mean free path--##
nuc_mfp = ((8 * nuc_diffusivity) / (np.pi * nuc_speed)) # m 

##--Calculate g coefficient--##
nuc_g = (2**(1/2) / (3 * nuc_diam * nuc_mfp)) * ((nuc_diam + nuc_mfp)**3 - (nuc_diam**2 + nuc_mfp**2)**(3/2)) - nuc_diam


##--Loop through dfs in diameter_dfs and calculate needed variables for each bin--##
##--Store in series starting at zero--##
coagulation_sink = pd.Series(0, index=aimms_time)  

for diameter, df in diameter_dfs.items():
    
    ##--Convert column diams from string to float--##
    mean_diameter = (float(diameter)) * 1E-9 # in m
    
    ##--Particle volume per diameter bin--##
    volume = (4/3) * np.pi * (mean_diameter / 2) ** 3 #m^3
    
    ##--Particle mass, assuming a density of 1--##
    mass = volume # kg
    
    ##--Particle speed--##
    speed = ((8 * k * temperature_series) / (np.pi * mass))**(1/2)
    
    ##--Reduced mass ratio--##
    z = mass/Mair
    
    ##--Collision cross section air + particle--##
    sigma = (mean_diameter + Dair) / 2
    
    ##--Extract Particle Concentration (first column in diameter_dfs)--##
    df['Particle_concentration'] = df.iloc[:, 0] / 1E-6 # converted to #/m^3
    
    ##--Estimate of mean free path (against collision with air) for use in slip correction--##
    mfp_estimate = 1/(np.pi * (1 + z)**(1/2) * Nair * sigma**2) # m
    
    ##--Knudsen number--##
    df['Knudsen_number'] = mfp_estimate / (mean_diameter / 2) # unitless
    
    ##--Cunningham slip correction in air--##
    slip = 1 + 2 * df['Knudsen_number'] * (2.514 + 0.800 * np.exp(-0.550 / df['Knudsen_number'])) # unitless
    
    ##--Particle diffusivity--##
    diffusivity = (k * temperature_series * nuc_slip / (3 * np.pi * dynam_viscosity * mean_diameter)) # m^2/s
    
    ##--Calculate mean free path of H2SO4 from molecular diameter--##
    df['mean_free_path'] = ((8 * diffusivity) / (np.pi * speed)) # m
    
    ##--Calculate g coefficient--##
    g = ((2**(1/2) / (3 * mean_diameter * df['mean_free_path'])) * ((mean_diameter + df['mean_free_path'])**3 
                                      - (mean_diameter**2 + df['mean_free_path']**2)**(3/2)) - mean_diameter)
    
    ##--Compute the coagulation kernel per bin--##
    df['Coagulation_kernel'] = (2 * np.pi * (nuc_diffusivity + diffusivity) * (nuc_diam + mean_diameter) * 
                          ((nuc_diam + mean_diameter) / (nuc_diam + mean_diameter + 2*(nuc_g**2 + g**2)**(1/2))
                           + (8 * (nuc_diffusivity + diffusivity)) / ((nuc_speed**2 + speed**2)**(1/2) * 
                                                                      (nuc_diam + mean_diameter)))**-1)
    
    ##--Calculate coagulation by multiplying kernel by particle concentration per bin--##
    df['Coagulation'] = df['Coagulation_kernel'] * df['Particle_concentration'] # s^-1
    
    ##--Sum the coagulation kernels across bins--##
    coagulation_sink += (df['Coagulation']).fillna(0)  

##--Populate series--##
coagulation_sink = pd.DataFrame({'Coagulation': coagulation_sink}) 

#####################################
##--Calculate std dev from zeroes--##        
#####################################

##--Pull datasets with zeros not filtered out--##
##--Worth it to do flight by flight or no?--##
CPC3_R1 = icartt.Dataset(r"C:\Users\repooley\REP_PhD\NETCARE2015\data\raw\CPC_R1\CPC3776_Polar6_20150408_R1_L2.ict")    
CPC10_R1 = icartt.Dataset(r'C:\Users\repooley\REP_PhD\NETCARE2015\data\raw\CPC_R1\CPC3772_Polar6_20150408_R1_L2.ict')
CPC3_R1_conc = CPC3_R1.data['conc']
CPC10_R1_conc = CPC10_R1.data['conc']

##--Isolate zero periods, setting conservative upper limit of 50--##
##--Numpy doesn't recognize -9999 as NaN, tell it to ignore these values--##
CPC3_zeros_c = CPC3_R1_conc[(CPC3_R1_conc < 50) & (CPC3_R1_conc != -9999)]
CPC10_zeros_c = CPC10_R1_conc[(CPC10_R1_conc < 50) & (CPC10_R1_conc != -99999)]

##--Calculate standard deviation of zeros--##
CPC3_sigma = np.std(CPC3_zeros_c, ddof=1)  # Use ddof=1 for sample standard deviation
CPC10_sigma = np.std(CPC10_zeros_c, ddof=1)

#############################
##--Propagate uncertainty--##
#############################

##--The ICARTT files for CPC instruments say 10% uncertainty of meas value - feels conservative for large counts!--##
##--Calculate the 3 sigma uncertainty for nucleating particles--##

T_error = 0.3 # K, constant
P_error = 100 + 0.0005*(pressure)

##--Use formula for multiplication/division--##
greater3nm_error = (CPC3_conc_aligned)*(((P_error)/(pressure))**2 + ((T_error)/(temperature))**2 + ((CPC3_sigma)/(CPC3_conc_aligned)))**(0.5)
greater10nm_error = (CPC10_conc_aligned)*(((P_error)/(pressure))**2 + ((T_error)/(temperature))**2 + ((CPC10_sigma)/(CPC10_conc_aligned)))**(0.5)

##--Use add/subtract forumula--##
nuc_error_3sigma = (((greater3nm_error)**2 + (greater10nm_error)**2)**(0.5))*3

#######################################
##--Filter to NPF and non-NPF times--##
#######################################

condensation_n_3_10 = pd.DataFrame({'Condensation': condensation_sink['Condensation_Sink'], 'Nucleation': n_3_10['6'],
                                 'LoD': nuc_error_3sigma})
condensation_npf = condensation_n_3_10['Condensation'][condensation_n_3_10['Nucleation'] > condensation_n_3_10['LoD']]
condensation_nonpf = condensation_n_3_10['Condensation'][condensation_n_3_10['Nucleation'] <= condensation_n_3_10['LoD']]
conden_df = {'NPF': condensation_npf, 'No NPF': condensation_nonpf}

coagulation_n_3_10 = pd.DataFrame({'Coagulation': coagulation_sink['Coagulation'], 'Nucleation': n_3_10['6'],
                                   'LoD': nuc_error_3sigma})
coagulation_npf = coagulation_n_3_10['Coagulation'][coagulation_n_3_10['Nucleation'] > coagulation_n_3_10['LoD']]
coagulation_nonpf = coagulation_n_3_10['Coagulation'][coagulation_n_3_10['Nucleation'] <= coagulation_n_3_10['LoD']]
coag_df = {'NPF':coagulation_npf, 'No NPF': coagulation_nonpf}

BC_n_3_10 = pd.DataFrame({'BC': BC_count_aligned, 'Nucleation': n_3_10['6'],
                                   'LoD': nuc_error_3sigma})
BC_npf = BC_n_3_10['BC'][BC_n_3_10['Nucleation'] > BC_n_3_10['LoD']]
BC_nonpf = BC_n_3_10['BC'][BC_n_3_10['Nucleation'] <= BC_n_3_10['LoD']]
BC_df = {'NPF':BC_npf, 'No NPF': BC_nonpf}

#############
##--Stats--##
#############

##--Counts--##
conden_npf_count = len(condensation_npf)
conden_nonpf_count = len(condensation_nonpf)
coag_npf_count = len(coagulation_npf)
coag_nonpf_count = len(coagulation_nonpf)
BC_npf_count = len(BC_npf)
BC_nonpf_count = len(BC_nonpf)

##--Statistical signficance for unpaired non-parametric data: Mann-Whitney U test--##
conden_npf_array = condensation_npf.dropna().to_numpy() # data should be in a list or array
conden_nonpf_array = condensation_nonpf.dropna().to_numpy()
U_conden, p_conden = mannwhitneyu(conden_npf_array, conden_nonpf_array)

##--Calculate Z-score--##
##--Referenced https://datatab.net/tutorial/mann-whitney-u-test--##
z_conden = (U_conden - conden_npf_count*conden_nonpf_count/2)/((conden_npf_count*
            conden_nonpf_count*(conden_npf_count + conden_nonpf_count + 1)/12)**(1/2))

##--Take absolute value of Z scores--##
z_conden = abs(z_conden)

##--Use Z-score to calculate rank biserial correlation, r--##
r_conden = z_conden/((conden_npf_count + conden_nonpf_count)**(1/2))

coag_npf_array = coagulation_npf.dropna().to_numpy()
coag_nonpf_array = coagulation_nonpf.dropna().to_numpy()
U_coag, p_coag = mannwhitneyu(coag_npf_array, coag_nonpf_array)

z_coag = (U_coag - coag_npf_count*coag_nonpf_count/2)/((coag_npf_count*
            coag_nonpf_count*(coag_npf_count + coag_nonpf_count + 1)/12)**(1/2))

z_coag = abs(z_coag)

r_coag = z_coag/((coag_npf_count + coag_nonpf_count)**(1/2))

BC_npf_array = BC_npf.dropna().to_numpy()
BC_nonpf_array = BC_nonpf.dropna().to_numpy()
U_BC, p_BC = mannwhitneyu(BC_npf_array, BC_nonpf_array)

z_BC = (U_BC - BC_npf_count*BC_nonpf_count/2)/((BC_npf_count*
            BC_nonpf_count*(BC_npf_count + BC_nonpf_count + 1)/12)**(1/2))

z_BC = abs(z_BC)

r_BC = z_BC/((BC_npf_count + BC_nonpf_count)**(1/2))

################
##--Plotting--##
################

##--Assign correct colors to high or low latitude flights--##
##--Separate flights by latitude--##
high_lat_flights = {'Flight2', 'Flight3', 'Flight4', 'Flight5', 'Flight6', 'Flight7'}

if flight in high_lat_flights: 
    palette = {'NPF':'#2f6794', 'No NPF':'#1e537e'}
    palette2 = {'NPF':'#4c88b8', 'No NPF':'#3A75A5'}
    palette3 = {'NPF':'#477D7C', 'No NPF':'#345B5A'}
else: 
    palette = {'NPF':'#C00000', 'No NPF':'#820000'}
    palette2 = {'NPF':'#EA1010', 'No NPF':'#9D0B0B'}
    palette3 = {'NPF': '#CC1616', 'No NPF': '#800E0E'}

fig, ax = plt.subplots(figsize = (4,6))
##--Cut=0 disallows interpolation beyond the data extremes--##
condensation_plot = sns.violinplot(data=conden_df, palette=palette,
                                   inner_kws={'whis_width': 0, 'solid_capstyle':'butt'}, ax=ax, cut=0)
ax.set(xlabel='')
ax.set(ylabel='Condensation Sink (S-1)')
ax.set(title=f"Condensation Sink - {flight.replace('Flight', 'Flight ')}")

##--Add text labels with N--##
plt.text(0.25, 0.12, "N={}".format(conden_npf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')
plt.text(0.63, 0.12, "N={}".format(conden_nonpf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')

##--Conditions for adding p values--##
if p_conden >= 0.05:
    plt.text(0.33, 0.855, f"p={p_conden:.4f},", transform=fig.transFigure, fontsize=10, color='dimgrey')
elif 0.05 > p_conden >= 0.0005:
    plt.text(0.33, 0.855, f"p={p_conden:.4f},", transform=fig.transFigure, fontsize=10, color='dimgrey')
elif p_conden < 0.0005: 
    plt.text(0.33, 0.855, "p<0.0005,", transform=fig.transFigure, fontsize=10, color='dimgrey')
    
##--Add r value next to p-value--##
plt.text(0.525, 0.855, f"r={r_conden:.3f}", transform=fig.transFigure, fontsize=10, color='dimgrey')
 
plt.savefig(f"{output_path}\\Sinks\condensation\conden_{flight}", dpi=600)

plt.show()

fig, ax = plt.subplots(figsize=(4,6))
coagulation_plot = sns.violinplot(data = coag_df, order=['NPF', 'No NPF'], 
                                  inner_kws={'whis_width': 0, 'solid_capstyle':'butt'}, palette=palette2, ax=ax, cut=0)
ax.set(xlabel='')
ax.set(ylabel='Coagulation Sink (S-1)')
ax.set(title=f"Coagulation Sink - {flight.replace('Flight', 'Flight ')}")

##--Add text labels with N--##
plt.text(0.25, 0.12, "N={}".format(coag_npf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')
plt.text(0.63, 0.12, "N={}".format(coag_nonpf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')

##--Conditions for adding p values--##
if p_coag >= 0.05:
    plt.text(0.33, 0.855, f"p={p_coag:.4f},", transform=fig.transFigure, fontsize=10, color='dimgrey')
elif 0.05 > p_coag >= 0.0005:
    plt.text(0.33, 0.855, f"p={p_coag:.4f},", transform=fig.transFigure, fontsize=10, color='dimgrey')
elif p_coag < 0.0005: 
    plt.text(0.33, 0.855, "p<0.0005,", transform=fig.transFigure, fontsize=10, color='dimgrey')
    
##--Add r value next to p-value--##
plt.text(0.525, 0.855, f"r={r_coag:.3f}", transform=fig.transFigure, fontsize=10, color='dimgrey')
 
plt.savefig(f"{output_path}\\Sinks\coagulation\coag_{flight}", dpi=600)

plt.show()

fig, ax = plt.subplots(figsize=(4,6))
BC_plot = sns.violinplot(data = BC_df, order=['NPF', 'No NPF'], 
                                  inner_kws={'whis_width': 0, 'solid_capstyle':'butt'}, palette=palette3, ax=ax, cut=0)
ax.set(xlabel='')
ax.set(ylabel='rBC Abundance (counts/cm\u00B3)')
ax.set(title=f"rBC Abundance - {flight.replace('Flight', 'Flight ')}")

##--Add text labels with N--##
plt.text(0.25, 0.12, "N={}".format(coag_npf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')
plt.text(0.63, 0.12, "N={}".format(coag_nonpf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')

##--Conditions for adding p values--##
if p_BC >= 0.05:
    plt.text(0.33, 0.855, f"p={p_BC:.4f},", transform=fig.transFigure, fontsize=10, color='dimgrey')
elif 0.05 > p_BC >= 0.0005:
    plt.text(0.33, 0.855, f"p={p_BC:.4f},", transform=fig.transFigure, fontsize=10, color='dimgrey')
elif p_BC < 0.0005: 
    plt.text(0.33, 0.855, "p<0.0005,", transform=fig.transFigure, fontsize=10, color='dimgrey')
    
##--Add r value next to p-value--##
plt.text(0.525, 0.855, f"r={r_BC:.3f}", transform=fig.transFigure, fontsize=10, color='dimgrey')
 
plt.savefig(f"{output_path}\\rBC\\rBC_{flight}", dpi=600)

plt.show()
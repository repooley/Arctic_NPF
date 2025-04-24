# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 16:15:02 2025

@author: repooley
"""

##--Sink calculations are currently too LOW by 1-2 orders of magnitude--##
##--Sinks currently populating with NaNs--##

import icartt
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker

#########################
##--Open ICARTT Files--##
#########################

##--Set the base directory to project folder--##
directory = r"C:\Users\repooley\REP_PhD\NETCARE2015\data"

##--Select flight (Flight2 thru Flight10)--##
##--NO UHSAS FILES FOR FLIGHT1--##
flight = "Flight2"

##--Define function that creates datasets from filenames--##
def find_files(directory, flight, partial_name):
    ##--flight data are stored in a folder called "raw"--##
    flight_dir = os.path.join(directory, "raw", flight)
    search_pattern = os.path.join(flight_dir, f"*{partial_name}*")
    return sorted(glob.glob(search_pattern))

##--Meterological data from AIMMS monitoring system--##
aimms = icartt.Dataset(find_files(directory, flight, "AIMMS_POLAR6")[0])

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

##--USHAS Data--##
UHSAS_time = UHSAS.data['time'] # seconds since midnight

##--Reindex UHSAS_time to aimms_time--##
if len(UHSAS_time) < len(aimms_time):
    # Pad with NaN to match AIMMS time
    padding_length = len(aimms_time) - len(UHSAS_time)
    ##--Name reindex time variable separately--##
    UHSAS_time_ri = np.append(UHSAS_time, [np.nan]*padding_length)

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

##--Rename with new column names--##
UHSAS_bins.columns = UHSAS_new_col_names

##--Add time, total_num to UHSAS_bins df--##
UHSAS_bins.insert(0, 'Time', UHSAS_time)

##--Align UHSAS_bins time to AIMMS time--##
UHSAS_bins_aligned = UHSAS_bins.set_index('Time').reindex(aimms_time)

##--Tabulate total count across all bins--##
UHSAS_total_num = UHSAS_bins_aligned.sum(axis=1, numeric_only=True)

##--Align the total count to AIMMS time separately--##
##--Convert total num to a df--##
UHSAS_total_num = pd.DataFrame({'Time' : UHSAS_time_ri, 'Total_count': UHSAS_total_num})
UHSAS_total_aligned = UHSAS_total_num.set_index('Time').reindex(aimms_time)

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

##--10 nm CPC data--##
CPC10_time = CPC10.data['time']
CPC10_conc = CPC10.data['conc'] # count/cm^3

##--2.5 nm CPC data--##
CPC3_time = CPC3.data['time']
CPC3_conc = CPC3.data['conc'] # count/cm^3

##################
##--UHSAS Data--##
##################

##--Convert to dataframe--##
#UHSAS_total_num = pd.DataFrame(UHSAS_total_num, columns = ['total_number_conc'])

##--ABSOLUTE COUNTS--##

##--Calculate dlogDp for UHSAS bins--##
UHSAS_dlogDp = pd.Series(
    data=np.log(UHSAS_upper_bound.values) - np.log(UHSAS_lower_bound.values),
    index=UHSAS_new_col_names  # same as UHSAS_particle_counts.columns
)

##--Get only particle count data (excluding 'Time')--##
UHSAS_particle_counts = UHSAS_bins[UHSAS_new_col_names]

##--De-Normalize counts by multiplying by dlogDp across all rows--##
UHSAS_denorm_counts = UHSAS_particle_counts.multiply(UHSAS_dlogDp, axis=1)

# --Assign time index BEFORE reindexing-- #
UHSAS_denorm_counts['Time'] = UHSAS_bins['Time'].values
UHSAS_denorm_counts = UHSAS_denorm_counts.set_index('Time')

##--Reindex to aimms_time--##
UHSAS_denorm_counts = UHSAS_denorm_counts.reindex(aimms_time)

##--Take out of STP--##
P_STP = 101325  # Pa
T_STP = 273.15  # K

##--Create empty list for UHSAS particles--##
UHSAS_abs_counts = []

for UHSAS, T, P in zip(UHSAS_denorm_counts.values, temperature, pressure):
    if np.isnan(T) or np.isnan(P):
        ##--Append with NaN if any input is NaN--##
        UHSAS_abs_counts.append([np.nan]*len(UHSAS))
    else:
        ##--Perform conversion if all inputs are valid--##
        corrected_UHSAS = UHSAS / (P_STP / P) / (T / T_STP)
        UHSAS_abs_counts.append(corrected_UHSAS)

##--Convert UHSAS particle list to a dataframe--##
UHSAS_abs_counts = pd.DataFrame(UHSAS_abs_counts, columns=UHSAS_denorm_counts.columns, index=aimms_time)

##--Calculate total count at ambient condition--##
UHSAS_total_num_abs = UHSAS_total_aligned.div(P_STP / pressure, axis=0).div(temperature / T_STP, axis=0)

##--NORMALIZED--##

##--Create df with UHSAS total counts--##
UHSAS_total_STP = pd.DataFrame({'Time': aimms_time, 'Total_count': UHSAS_total_aligned['Total_count']})

##--Reindex UHSAS_total df to AIMMS time--##
UHSAS_total_aligned = UHSAS_total_STP.set_index('Time').reindex(aimms_time)

################
##--OPC Data--##
################

##--ABSOLUTE COUNTS--##

##--OPC samples every six seconds. Most rows are NaN--##
##--Forward-fill NaN values to propagate last valid reading--##
##--Limit forward filling to 5 NaN rows--##
OPC_bins_filled = OPC_bins_aligned.ffill(limit=5)

##--NORMALIZED--##

##--Calculate dlogDp for each bin in numpy array--##
dlogDp = np.log(OPC_upper_bound.values) - np.log(OPC_lower_bound.values)

##--Get only particle count data (excluding 'Time')--##
OPC_particle_counts = OPC_bins_filled.loc[:, OPC_new_col_names]

##--Normalize counts by dividing by dlogDp across all rows--##
OPC_dNdlogDp = OPC_bins_filled.divide(dlogDp, axis=1)

##--Convert to STP!--##
P_STP = 101325  # Pa
T_STP = 273.15  # K

##--Create empty list for OPC particles--##
OPC_conc_STP = []

for OPC, T, P in zip(OPC_dNdlogDp.values, temperature, pressure):
    if np.isnan(T) or np.isnan(P):
        ##--Append with NaN if any input is NaN--##
        OPC_conc_STP.append([np.nan]*len(OPC))
    else:
        ##--Perform conversion if all inputs are valid--##
        corrected_OPC = OPC * (P_STP / P) * (T / T_STP)
        OPC_conc_STP.append(corrected_OPC)

##--Convert back to DataFrame with same columns and index--##
OPC_conc_STP = pd.DataFrame(OPC_conc_STP, columns=OPC_dNdlogDp.columns, index=OPC_dNdlogDp.index)

######################
##--Calc N(2.5-10)--##
######################

##--Put N(2.5-10) bin center in a df--##
n_3_10_center = pd.DataFrame({'bin_avg': [6.5]}) # Approximate mean of 2.5 and 10

##--ABSOLUTE COUNTS--##

##--Make CPC3 df and set index to CPC3 time--##
CPC3_df = pd.DataFrame({'time': CPC3_time, 'conc': CPC3_conc}).set_index('time')
##--Make a new df reindexed to aimms_time. Populate with CPC3 conc--##
CPC3_conc_aligned = CPC3_df.reindex(aimms_time)['conc']

##--Make CPC10 df and set index to CPC10 time--##
CPC10_df = pd.DataFrame({'time': CPC10_time, 'conc': CPC10_conc}).set_index('time')
##--Make a new df reindexed to aimms_time. Populate with CPC10 conc--##
CPC10_conc_aligned = CPC10_df.reindex(aimms_time)['conc']

##--Create a pandas dataframe for CPC data--##
CPC_df = pd.DataFrame({'Altitude' : altitude, 'Latitude': latitude, 'CPC3_conc':CPC3_conc_aligned, 'CPC10_conc': CPC10_conc_aligned})

##--Calculate N(2.5-10)--##
nuc_particles = (CPC_df['CPC3_conc'] - CPC_df['CPC10_conc'])

##--Change calculated particle counts less than zero to NaN--##
nuc_particles = np.where(nuc_particles >= 0, nuc_particles, np.nan)

##--Create a df for N(2.5-10)--##
n_3_10 = pd.DataFrame({'time': aimms_time, '6.25': nuc_particles}).set_index('time')

##--NORMALIZED--##

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
CPC_STP_df = pd.DataFrame({'Altitude': altitude, 'Latitude': latitude, 'CPC3_conc':CPC3_conc_STP, 'CPC10_conc': CPC10_conc_STP})

##--Calculate N3-10 particles--##
nuc_particles_STP = (CPC_STP_df['CPC3_conc'] - CPC_STP_df['CPC10_conc'])

##--Change calculated particle counts less than zero to NaN--##
nuc_particles_STP = np.where(nuc_particles_STP >= 0, nuc_particles_STP, np.nan)

##--Create a dataframe for N 2.5-10--##
n_3_10_STP = pd.DataFrame({'time': aimms_time, '6.25': nuc_particles_STP}).set_index('time')

#####################
##--Calc N(10-89)--##
#####################

##--Put N(10-85) bin center in a df--##
n_10_89_center = pd.DataFrame({'bin_avg': [49.5]})

##--ABSOLUTE COUNTS--##

##--Create df with CPC10 counts and set index to time--##
CPC10_counts = pd.DataFrame({'Time': aimms_time, 'Counts': CPC10_conc_aligned}).set_index('Time')

##--Calculate particles below UHSAS lower cutoff--##
n_10_89 = (CPC10_counts['Counts'] - UHSAS_total_num_abs['Total_count'])

##--Change calculated particle counts less than zero to NaN--##
n_10_89 = np.where(n_10_89 >= 0, n_10_89, np.nan)

##--Convert n_10_89 to a df--##
n_10_89 = pd.DataFrame({'49.5': n_10_89, 'time': aimms_time}).set_index('time')

##--NORMALIZED--##

##--Create df with CPC10 counts and set index to time--##
CPC10_STP_counts = pd.DataFrame({'Time':aimms_time, 'Counts':CPC10_conc_STP}).set_index('Time')

##--Calculate particles below UHSAS lower cutoff--##
n_10_89_STP = (CPC10_STP_counts['Counts'] - UHSAS_total_aligned['Total_count'])

##--Change calculated particle counts less than zero to NaN--##
n_10_89_STP = np.where(n_10_89_STP >= 0, n_10_89_STP, np.nan)

##--Convert n_10_89 to a df--##
n_10_89_STP = pd.DataFrame({'49.5': n_10_89_STP, 'time':aimms_time}).set_index('time')

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
bin_centers = pd.concat([n_3_10_center, n_10_89_center, UHSAS_bin_center, OPC_bin_center], axis=0).reset_index(drop=True)

##--ABSOLUTE COUNTS--##

##--Place all binned data in a single df--##
all_bins_aligned = pd.concat([n_3_10, n_10_89, UHSAS_abs_counts, OPC_bins_filled], axis=1)
total_particle_count = all_bins_aligned.sum(axis=1, numeric_only =True)

##--Create a dictionary to store each column as a separate dataframe, col names are keys--##
diameter_dfs = {col: pd.DataFrame({col: all_bins_aligned[col]}) for col in all_bins_aligned.columns}

##--NORMALIZED--##

##--Place all binned data in a single df--##
all_bins_aligned_STP = pd.concat([n_3_10_STP, n_10_89_STP, UHSAS_bins, OPC_conc_STP], axis=1)
total_particle_count_STP = all_bins_aligned_STP.sum(axis=1, numeric_only=True) 


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
cs_sum = pd.Series(0, index=aimms_time)  # Temporary storage for summing CS contributions

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
    
    ##--Sum the contributions across bins--##
    cs_sum += df['CS_contribution']  # Accumulate per bin

condensation_sink = pd.DataFrame(index=aimms_time, columns=['Condensation_Sink'])
condensation_sink['Condensation_Sink'] = 0 # initialize to zero instead of NaN for proper summing index later  

##--Final multiplication by 2 * π * diffusion coefficient--##
condensation_sink['Condensation_Sink'] = 2 * 3.14159 * df['Diffusion_coefficient'] * cs_sum

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
coag_sum = pd.Series(0, index=aimms_time)  # Temporary storage for summing CS contributions

for diameter, df in diameter_dfs.items():
    
    ##--Convert column diams from string to float--##
    mean_diameter = (float(diameter)) * 1E-9 # in m
    
    ##--Particle volume per diameter bin--##
    volume = (4/3) * np.pi * (nuc_diam / 2) ** 3 #m^3
    
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
    coag_sum += df['Coagulation']  # Accumulate per bin

##--Create dataframe and populate with coag_sum series--##    
coagulation_sink = pd.DataFrame(index=aimms_time, columns=['Coagulation'])
coagulation_sink['Coagulation'] = coag_sum

################
##--Plotting--##
################

##--Base plot uses the STP particle data--##

##--Apply rolling average to all particle data--##
all_bins_smoothed = all_bins_aligned_STP.rolling(window=30, min_periods=1, center=True).mean()

##--Numpy array expected by pcolormesh--##
all_conc = all_bins_smoothed.to_numpy().T 

##--Use pcolormesh which is more flexible than imshow--## 
fig, ax1 = plt.subplots(figsize=(12, 8))
c = ax1.pcolormesh(time_edges, combined_bin_edges, all_conc, shading='auto', cmap='viridis')

##--Labels and colorbar--##
cb = plt.colorbar(c, ax=ax1, location='bottom', pad=0.1, shrink=0.65)
cb.set_label(label='Normalized Particle Concentration (dN/dlogDp) [scm⁻³]', fontsize=14)
cb.ax.tick_params(labelsize=14)

# Optional adjustments
ax1.set_title(f"Optical Data Time Series - {flight.replace('Flight', 'Flight ')}", fontsize=20, pad=20)
ax1.set_xlim([aimms_time.min(), aimms_time.max()])
#ax1.set_ylim([bin_center.min(), bin_center.max()])
ax1.set_yscale('log')
custom_ticks = [10, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
##--Apply custom ticks--##
ax1.yaxis.set_major_locator(ticker.FixedLocator(custom_ticks))

# Optional: format them normally (just numbers)
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
ax1.tick_params(axis='y', labelsize=14)
ax1.tick_params(axis='x', labelsize=14)

ax1.set_xlabel('Time (seconds since midnight UTC)', fontsize=14)
ax1.set_ylabel('Log of Particle Diameter (nm)', fontsize=14)

##--Plot the coagulation sink with a separate y-axis--##
ax2 = ax1.twinx()  # Create a second y-axis
window_size = 100  # Smoothing window size 
coagulation_sink['Coagulation_smoothed'] = coagulation_sink['Coagulation'].rolling(window=window_size, min_periods=1, center=True).mean()

ax2.plot(coagulation_sink.index, coagulation_sink['Coagulation_smoothed'], color='gray', label='Coagulation Sink', linewidth=2)

# Set labels for the secondary y-axis
ax2.set_ylabel('Sinks (S-1)', fontsize=14)
ax2.tick_params(axis='y', labelsize=14)

# Optional: Add a legend for the coagulation sink line
ax2.legend(loc='upper left', fontsize=12)

# Apply smoothing for condensation sink (same method as coagulation sink)
condensation_sink['Condensation_Sink_smoothed'] = condensation_sink['Condensation_Sink'].rolling(window=window_size, min_periods=1, center=True).mean()

# Plot the condensation sink on the same axis as the coagulation sink
ax2.plot(condensation_sink.index, condensation_sink['Condensation_Sink_smoothed'], color='black', label='Condensation Sink', linewidth=2)

# Optional: Add a legend for the condensation sink line
ax2.legend(loc='upper right', fontsize=12)

plt.tight_layout()
plt.show()

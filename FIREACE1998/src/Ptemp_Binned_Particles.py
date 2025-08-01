# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 11:26:14 2025

@author: repooley
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

###################
##--User inputs--##
###################

##--Set the base directory to project folder--##
directory = r"C:\Users\repooley\REP_PhD\Arctic_NPF\FIREACE1998\data"

##--Select flight (F01 thru F18)--##
##--NO 1 hz data for flights 4,5,6 currently--##
flight = "Flight1"

##--Base output path in directory--##
output_path = r"C:\Users\repooley\REP_PhD\Arctic_NPF\FIRACE1998\data\processed\VerticallyBinnedData"

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

#%%
#############################
##--Propagate uncertainty--##
#############################

##--The CPC3 instrument nominally has +-10% uncertainty, but with the flow error it's higher--##
##--Going to guess 15% for now--##
CPC3_sigma = 0.15*(CPC3_data)

##--The CPC10 instrument has a +-10% uncertainty--##
CPC10_sigma = 0.1*(CPC3_data)

##--Not sure of what instruments were onboard Convair 580, using values from NETCARE instruments--##
T_error = 0.3 # K, constant
P_error = 100 + 0.0005*(pressure)

##--Use formula for mult/div to compute error after converting to STP--##
greater3nm_error = (CPC3_data)*(((P_error)/(pressure))**2 + ((T_error)/(temperature))**2 + ((CPC3_sigma)/(CPC3_data)))**(0.5)
greater10nm_error = (CPC10_data)*(((P_error)/(pressure))**2 + ((T_error)/(temperature))**2 + ((CPC10_sigma)/(CPC10_data)))**(0.5)

##--Use add/subtract forumula to compute 3sigma error--##
nuc_error_3sigma = (((greater3nm_error)**2 + (greater10nm_error)**2)**(0.5))*3

##--nuc_error_3sigma still has a time index, reset to integer to align--##
df['nuc_error_3sigma'] = nuc_error_3sigma

#%%
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

df['ptemp'] = potential_temp
    
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

#%%
################
##--PLOTTING--##
################

##--Creates figure with 3 horizontally stacked subplots sharing a y-axis--##
fig, axs = plt.subplots(1, 3, figsize=(9, 6), sharey=True)

##--First subplot: 10+ nm Particles vs Altitude--##

##--Averaged data in each bin is plotted against bin center--##
axs[0].plot(binned_df['CPC10_conc_center'], binned_df['ptemp_center'], color='maroon')
##--Range is given by filling between data minimum and maximum for each bin--##
axs[0].fill_betweenx(binned_df['ptemp_center'], binned_df['CPC10_conc_min'], 
                     binned_df['CPC10_conc_max'], color='indianred', alpha=0.25)
axs[0].fill_betweenx(binned_df['ptemp_center'], binned_df['CPC10_conc_25th'],
                    binned_df['CPC10_conc_75th'], color='indianred', alpha=0.7)
axs[0].set_ylabel('Potential Temperature (K)', fontsize=12)
axs[0].set_xlabel('Counts/cm\u00b3')
axs[0].set_title('N \u2265 10 nm')
#axs[0].set_xlim(-50, 1500)

##--Second subplot: 2.5+ nm Particles vs Altitude--##
axs[1].plot(binned_df['CPC3_conc_center'], binned_df['ptemp_center'], color='saddlebrown')
axs[1].fill_betweenx(binned_df['ptemp_center'], binned_df['CPC3_conc_min'], 
                     binned_df['CPC3_conc_max'], color='sandybrown', alpha=0.25)
axs[1].fill_betweenx(binned_df['ptemp_center'], binned_df['CPC3_conc_25th'],
                    binned_df['CPC3_conc_75th'], color='sandybrown', alpha=1)
axs[1].set_title('N \u2265 2.5 nm')
axs[1].set_xlabel('Counts/cm\u00b3')
#axs[1].set_xlim(-50, 2000)

##--Third subplot: Nuc particles vs Altitude--##
axs[2].plot(binned_df['nuc_particles_center'], binned_df['ptemp_center'], color='darkslategray')
axs[2].fill_betweenx(binned_df['ptemp_center'], binned_df['nuc_particles_min'], 
                     binned_df['nuc_particles_max'], color='cadetblue', alpha=0.25)
axs[2].fill_betweenx(binned_df['ptemp_center'], binned_df['nuc_particles_25th'],
                    binned_df['nuc_particles_75th'], color='cadetblue', alpha=1)

##--Plot uncertainty as its own trace--##
axs[2].plot(binned_df['nuc_error_center'], binned_df['ptemp_center'], color='crimson', 
            linestyle='dashed', label='3$\sigma$ \nuncertainty')

axs[2].legend(loc='lower right')

##--Subscript 3-10--##
axs[2].set_title('$N_{2.5-10}$')
axs[2].set_xlabel('Counts/cm\u00b3')
#axs[2].set_xlim(-50, 2000)

##--Use f-string to embed flight # variable in plot title--##
plt.suptitle(f"FIRE-ACE Vertical Particle Count Profile - {flight.replace('Flight', 'Flight ')}", fontsize=16)

##--Adjusts layout to prevent overlapping--## 
plt.tight_layout(rect=[0, -0.02, 1, 0.99])

##--Use f-string to save file with flight# appended--##
#output_path = f"{output_path}\\{flight}"
#plt.savefig(output_path, dpi=600, bbox_inches='tight') 

plt.show()
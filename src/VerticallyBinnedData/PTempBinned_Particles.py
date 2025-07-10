# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 11:40:48 2025

@author: repooley
"""

import icartt
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

###################
##--User inputs--##
###################

##--Set the base directory to project folder--##
directory = r"C:\Users\repooley\REP_PhD\NETCARE2015\data"

##--Select flight (Flight1 thru Flight10)--##
flight = "Flight10" 

##--Bin data are in a CSV file--##
UHSAS_bins = pd.read_csv(r"C:\Users\repooley\REP_PhD\NETCARE2015\data\raw\NETCARE2015_UHSAS_bins.csv")

##--Pull datasets with zeros not filtered out--##
CPC3_R1 = icartt.Dataset(r"C:\Users\repooley\REP_PhD\NETCARE2015\data\raw\CPC_R1\CPC3776_Polar6_20150408_R1_L2.ict")    
CPC10_R1 = icartt.Dataset(r'C:\Users\repooley\REP_PhD\NETCARE2015\data\raw\CPC_R1\CPC3772_Polar6_20150408_R1_L2.ict')

##--Base output path in directory--##
output_path = r"C:\Users\repooley\REP_PhD\NETCARE2015\data\processed\PTempBinnedData\Particle"

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

##--CPC data--##
CPC10 = icartt.Dataset(find_files(directory, flight, 'CPC3772')[0])
CPC3 = icartt.Dataset(find_files(directory, flight, 'CPC3776')[0])

#################
##--Pull data--##
#################

##--AIMMS Data--##
altitude = aimms.data['Alt'] # in m
temperature = aimms.data['Temp'] + 273.15 # in K
pressure = aimms.data['BP']
aimms_time =aimms.data['TimeWave']

##--Black carbon--##
BC_count = SP2.data['BC_numb_concSTP'] # in STP

##--10 nm CPC data--##
CPC10_time = CPC10.data['time']
CPC10_conc = CPC10.data['conc'] # count/cm^3

##--2.5 nm CPC data--##
CPC3_time = CPC3.data['time']
CPC3_conc = CPC3.data['conc'] # count/cm^3

##--USHAS Data--##
UHSAS_time = UHSAS.data['time'] # seconds since midnight
##--Total count is computed for N > 85 nm--##
UHSAS_total_num = UHSAS.data['total_number_conc'] # particles/cm^3

##################
##--Align data--##
##################

##--Establish AIMMS start/stop times--##
aimms_end = aimms_time.max()
aimms_start = aimms_time.min()

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

##--Handle CPC3 data with different start/stop times than AIMMS--##
CPC3_time = CPC3.data['time']

##--Trim CPC3 data if it starts before AIMMS--##
if CPC3_time.min() < aimms_start:
    mask_start = CPC3_time >= aimms_start
    CPC3_time = CPC3_time[mask_start]
    CPC3_conc = CPC3_conc[mask_start]
    
##--Append CPC3 data with NaNs if it ends before AIMMS--##
if CPC3_time.max() < aimms_end: 
    missing_times = np.arange(CPC3_time.max()+1, aimms_end +1)
    CPC3_time = np.concatenate([CPC3_time, missing_times])
    CPC3_conc = np.concatenate([CPC3_conc, [np.nan]*len(missing_times)])

##--Create a DataFrame for CPC3 data and reindex to AIMMS time, setting non-overlapping times to nan--##
CPC3_df = pd.DataFrame({'time': CPC3_time, 'conc': CPC3_conc})
CPC3_aligned = CPC3_df.set_index('time').reindex(aimms_time)
CPC3_aligned['conc']=CPC3_aligned['conc'].where(CPC3_aligned.index.isin(aimms_time), np.nan)
CPC3_conc_aligned = CPC3_aligned['conc']

##--Handle CPC10 data with different start/stop times than AIMMS--##
CPC10_time = CPC10.data['time']

##--Trim CPC10 data if it starts before AIMMS--##
if CPC10_time.min() < aimms_start:
    mask_start = CPC10_time >= aimms_start
    CPC10_time = CPC10_time[mask_start]
    CPC10_conc = CPC10_conc[mask_start]
    
##--Append CPC10 data with NaNs if it ends before AIMMS--##
if CPC10_time.max() < aimms_end: 
    missing_times = np.arange(CPC10_time.max()+1, aimms_end +1)
    CPC10_time = np.concatenate([CPC10_time, missing_times])
    CPC10_conc = np.concatenate([CPC10_conc, [np.nan]*len(missing_times)])

##--Create a DataFrame for CPC10 data and reindex to AIMMS time, setting non-overlapping times to nan--##
CPC10_df = pd.DataFrame({'time': CPC10_time, 'conc': CPC10_conc})
CPC10_aligned = CPC10_df.set_index('time').reindex(aimms_time)
CPC10_aligned['conc']=CPC10_aligned['conc'].where(CPC10_aligned.index.isin(aimms_time), np.nan)
CPC10_conc_aligned = CPC10_aligned['conc']

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

######################
##--Calc N(2.5-10)--##
######################

##--Convert to STP!--##
P_STP = 101325  # Pa
T_STP = 273.15  # K

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

##--Creates a Pandas dataframe for particle data--##
df = pd.DataFrame({'PTemp': potential_temp, 'BC_count': BC_count_aligned, 
                   'CPC3_conc':CPC3_conc_STP, 'CPC10_conc': CPC10_conc_STP})

##--Calculate N3-10 particles--##
nuc_particles = (df['CPC3_conc'] - df['CPC10_conc'])

##--Change calculated particle counts less than zero to NaN--##
nuc_particles = np.where(nuc_particles >= 0, nuc_particles, np.nan)

##--Add nucleating particles to df--##
df['nuc_particles'] = nuc_particles

#####################
##--Calc N(10-89)--##
#####################

##--Create df with UHSAS total counts--##
UHSAS_total = pd.DataFrame({'Time': UHSAS_time, 'Total_count': UHSAS_total_num})

##--Reindex UHSAS_total df to AIMMS time--##
UHSAS_total_aligned = UHSAS_total.set_index('Time').reindex(aimms_time)

##--Create df with CPC10 counts and set index to time--##
CPC10_counts = pd.DataFrame({'Time':aimms_time, 'Counts':CPC10_conc_STP}).set_index('Time')

##--Calculate particles below UHSAS lower cutoff--##
n_10_89 = (CPC10_counts['Counts'] - UHSAS_total_aligned['Total_count'])

##--Change calculated particle counts less than zero to NaN--##
n_10_89 = np.where(n_10_89 >= 0, n_10_89, np.nan)

##--Add 10-89 nm particles to the dataframe--##
df['n_10_89'] = n_10_89

#####################################
##--Calculate std dev from zeroes--##        
#####################################

##--Pull CPC data from R1 data--##
CPC3_R1_conc = CPC3_R1.data['conc']
CPC10_R1_conc = CPC10_R1.data['conc']

##--Isolate zero periods, setting conservative upper limit of 50 counts--##
##--Numpy doesn't recognize -9999 as NaN, tell it to ignore these values--##
CPC3_zeros_c = CPC3_R1_conc[(CPC3_R1_conc < 50) & (CPC3_R1_conc != -9999)]
CPC10_zeros_c = CPC10_R1_conc[(CPC10_R1_conc < 50) & (CPC10_R1_conc != -99999)]

##--Calculate standard deviation of zeros--##
CPC3_sigma = np.std(CPC3_zeros_c, ddof=1)  # Use ddof=1 for sample standard deviation
CPC10_sigma = np.std(CPC10_zeros_c, ddof=1)

##--Isolate zero periods, setting conservative upper limit of 100 counts--##
##--Numpy doesn't recognize -9999 as NaN, tell it to ignore these values--##
##--Need to set to 100 to see any data, but is this a good idea?--##
UHSAS_zeros_c = UHSAS_total_num[(UHSAS_total_num < 100) & (UHSAS_total_num != -9999)]

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

##--Add uncertainty to the df--##
df['nuc_error_3sigma'] = pd.Series(nuc_error_3sigma)#.reset_index(drop=True)

###############
##--BINNING--##
###############

##--Define number of bins here--##
num_bins = 124

##--Compute the minimum and maximum altitude, ignoring NaNs--##
min_ptemp = df['PTemp'].min(skipna=True)
max_ptemp = df['PTemp'].max(skipna=True)

##--Create bin edges from min_alt to max_alt--##
bin_edges = np.linspace(min_ptemp, max_ptemp, num_bins + 1)

##--Pandas 'cut' splits altitude data into specified number of bins--##
df['PTemp_bin'] = pd.cut(df['PTemp'], bins=bin_edges)

##--Group variables into each altitude bin--## 
##--Observed=false shows all bins, even empty ones--##
binned_df = df.groupby('PTemp_bin', observed=False).agg(
    
   ##--Aggregate data by mean, min, and max--##
    PTemp_center=('PTemp', 'median'), 
    BC_center=('BC_count', 'median'), 
    BC_min=('BC_count', 'min'),
    BC_max=('BC_count', 'max'),
    BC_25th=('BC_count', lambda x: x.quantile(0.25)),
    BC_75th=('BC_count', lambda x: x.quantile(0.75)),
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
    n_10_89_center=('n_10_89', 'median'), 
    n_10_89_min=('n_10_89', 'min'),
    n_10_89_max=('n_10_89', 'max'), 
    n_10_89_25th=('n_10_89', lambda x: x.quantile(0.25)),
    n_10_89_75th=('n_10_89', lambda x: x.quantile(0.75)),
    
    ##--Bin the uncertainty of nucleating particles--##
    nuc_error_center=('nuc_error_3sigma', 'median')  
    
    ##--Reset the index so Altitude_bin is just a column--##
).reset_index()

################
##--PLOTTING--##
################

##--Creates figure with 2 horizontally stacked subplots sharing a y-axis--##
fig, axs = plt.subplots(1, 5, figsize=(15, 6), sharey=True)

##--First subplot: 10+ nm Particles vs PTemp--##

##--Averaged data in each bin is plotted against bin center--##
axs[0].plot(binned_df['CPC10_conc_center'], binned_df['PTemp_center'], color='maroon')
##--Range is given by filling between data minimum and maximum for each bin--##
axs[0].fill_betweenx(binned_df['PTemp_center'], binned_df['CPC10_conc_min'], 
                     binned_df['CPC10_conc_max'], color='indianred', alpha=0.25)
axs[0].fill_betweenx(binned_df['PTemp_center'], binned_df['CPC10_conc_25th'],
                    binned_df['CPC10_conc_75th'], color='indianred', alpha=0.7)
axs[0].set_ylabel('Potential Temperature (K)', fontsize=12)
axs[0].set_title('N \u2265 10 nm')
axs[0].set_xlim(-50, 2000)

##--Add dashed horizontal lines for the polar dome boundaries--##
##--Boundaries are defined from Bozem et al 2019 (ACP)--##
axs[0].axhline(y=285, color='k', linestyle='--', linewidth=1)
axs[0].axhline(y=299, color='k', linestyle='--', linewidth=1)

##--Add text labels on the left-hand side within the plot area--##
##--Compute midpoints for label placement--##
polar_dome_label = 282
marginal_polar_dome_label = 288
x_text = axs[0].get_xlim()[0] + 1050  # offset from left edge

axs[0].text(x_text, polar_dome_label, 'Polar Dome',
        rotation=0, fontsize=10, color='k',
        verticalalignment='center', horizontalalignment='left')
axs[0].text(x_text, marginal_polar_dome_label, 'Marginal Dome',
        rotation=0, fontsize=10, color='k',
        verticalalignment='center', horizontalalignment='left')

##--Second subplot: 2.5+ nm Particles vs PTemp--##
axs[1].plot(binned_df['CPC3_conc_center'], binned_df['PTemp_center'], color='saddlebrown')
axs[1].fill_betweenx(binned_df['PTemp_center'], binned_df['CPC3_conc_min'], 
                     binned_df['CPC3_conc_max'], color='sandybrown', alpha=0.25)
axs[1].fill_betweenx(binned_df['PTemp_center'], binned_df['CPC3_conc_25th'],
                    binned_df['CPC3_conc_75th'], color='sandybrown', alpha=1)
axs[1].set_title('N \u2265 2.5 nm')
axs[1].set_xlim(-50, 3400)

axs[1].axhline(y=285, color='k', linestyle='--', linewidth=1)
axs[1].axhline(y=299, color='k', linestyle='--', linewidth=1)

##--Third subplot: Nuc particles vs PTemp--##
axs[2].plot(binned_df['nuc_particles_center'], binned_df['PTemp_center'], color='darkslategray')
axs[2].fill_betweenx(binned_df['PTemp_center'], binned_df['nuc_particles_min'], 
                     binned_df['nuc_particles_max'], color='cadetblue', alpha=0.25)
axs[2].fill_betweenx(binned_df['PTemp_center'], binned_df['nuc_particles_25th'],
                    binned_df['nuc_particles_75th'], color='cadetblue', alpha=1)

##--Plot uncertainty as its own trace--##
axs[2].plot(binned_df['nuc_error_center'], binned_df['PTemp_center'], color='crimson', linestyle='dashed', label='3$\sigma$ Uncertainty', zorder=4)

##--Subscript 3-10--##
axs[2].set_title('$N_{2.5-10}$')
axs[2].set_xlim(-50, 1900)

axs[2].axhline(y=285, color='k', linestyle='--', linewidth=1)
axs[2].axhline(y=299, color='k', linestyle='--', linewidth=1)

axs[2].legend()

##--Fourth subplot: 10-89 nm particles vs Altitude--##
axs[3].plot(binned_df['n_10_89_center'], binned_df['PTemp_center'], color='darkcyan')
axs[3].fill_betweenx(binned_df['PTemp_center'], binned_df['n_10_89_min'], 
                     binned_df['n_10_89_max'], color='turquoise', alpha=0.25)
axs[3].fill_betweenx(binned_df['PTemp_center'], binned_df['n_10_89_25th'],
                    binned_df['n_10_89_75th'], color='mediumturquoise', alpha=1)

axs[3].axhline(y=285, color='k', linestyle='--', linewidth=1)
axs[3].axhline(y=299, color='k', linestyle='--', linewidth=1)

##--Subscript 10-89--##
axs[3].set_title('$N_{10-89}$')
axs[3].set_xlim(-50, 2000)

##--Fifth subplot: rBC counts--##
axs[4].plot(binned_df['BC_center'], binned_df['PTemp_center'], color='steelblue')
axs[4].fill_betweenx(binned_df['PTemp_center'], binned_df['BC_min'], 
                     binned_df['BC_max'], color='skyblue', alpha=0.3)
axs[4].fill_betweenx(binned_df['PTemp_center'], binned_df['BC_25th'],
                    binned_df['BC_75th'], color='skyblue', alpha=1)

axs[4].axhline(y=285, color='k', linestyle='--', linewidth=1)
axs[4].axhline(y=299, color='k', linestyle='--', linewidth=1)

axs[4].set_title('rBC Counts')
#axs[4].set_xlim(-50, 2000)

##--Add one common x-axis label--##
fig.supxlabel('Particle Concentration (counts/cm\u00b3)', fontsize=12)

##--Use f-string to embed flight # variable in plot title--##
plt.suptitle(f"Vertical Particle Count Profiles - {flight.replace('Flight', 'Flight ')}", fontsize=16)

##--Adjusts layout to prevent overlapping--## 
plt.tight_layout(rect=[0, -0.02, 1, 0.99])

##--Use f-string to save file with flight# appended--##
output_path = f"{output_path}\\CPC_Data_{flight}"
plt.savefig(output_path, dpi=600, bbox_inches='tight') 

plt.show()
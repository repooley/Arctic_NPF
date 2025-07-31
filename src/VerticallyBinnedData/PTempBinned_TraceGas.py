# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:16:18 2025

@author: repooley
"""

import os
import glob
import icartt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

###################
##--User inputs--##
###################

##--Set the base directory to project folder--##
directory = r"C:\Users\repooley\REP_PhD\NETCARE2015\data"

##--Select flight (Flight1 thru Flight10)--##
flight = "Flight10" 

##--Base output path in directory--##
output_path = r"C:\Users\repooley\REP_PhD\NETCARE2015\data\processed\PTempBinnedData\TraceGas"

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

##--Trace gases--##
CO = icartt.Dataset(find_files(directory, flight, "CO_POLAR6")[0])
CO2 = icartt.Dataset(find_files(directory, flight, "CO2_POLAR6")[0])
H2O = icartt.Dataset(find_files(directory, flight, "H2O_POLAR6")[0])

##--Flight 2 has multiple ozone files requiring special handling--##
O3_files = find_files(directory, flight, "O3_")
if len(O3_files) == 0:
    raise FileNotFoundError("No O3 files found.")
elif len(O3_files) == 1 or flight != "Flight2": 
    O3 = icartt.Dataset(O3_files[0])
    O3_2 = None
##--Special case for Flight 2--##
else: 
    O3 = icartt.Dataset(O3_files[0])
    O3_2 = icartt.Dataset(O3_files[1])

#################
##--Pull data--##
#################

##--AIMMS Data--##
altitude = aimms.data['Alt'] # in m
aimms_time =aimms.data['TimeWave'] # seconds since midnight
temperature = aimms.data["Temp"] + 273.15 # in K
pressure = aimms.data['BP'] # in Pa

##--Black carbon--##
BC_mass = SP2.data['BC_mass_concSTP'] # in STP

##--Trace Gas Data--##
CO_conc = CO.data['CO_ppbv']
CO2_conc = CO2.data['CO2_ppmv']
H2O_conc = H2O.data['H2O_ppmv']
##--Put O3 data in list to make concatenation easier--##
O3_starttime = list(O3.data['Start_UTC'])
O3_conc = list(O3.data['O3'])

##--Check for O3_2 data and stich to end of O3 if populated--##
if O3_2 is not None:
    O3_starttime += list(O3_2.data['Start_UTC'])
    O3_conc += list(O3_2.data['O3'])

##################
##--Align time--##
##################

##--Arbitary reference date for datetime conversion--##
reference_date = pd.to_datetime('2015-01-01')

##--O3 data: addressing different data resolution compared to AIMMS--##

##--Convert O3_starttime to a datetime object--##
O3_starttime_dt = pd.to_datetime(O3_starttime, unit='s', origin=reference_date)

##--Calculate the seconds since midnight--##
O3_seconds_since_midnight = O3_starttime_dt.hour * 3600 + O3_starttime_dt.minute * 60 + O3_starttime_dt.second

##--Create O3 dataframe--##
O3_df = pd.DataFrame({'Time_UTC': O3_seconds_since_midnight,'O3': O3_conc})

##--Reindex O3 data to AIMMS time and set non-overlapping time values to NaN--##
O3_aligned = O3_df.set_index('Time_UTC').reindex(aimms_time)
O3_aligned['O3'] = O3_aligned['O3'].where(O3_aligned.index.isin(aimms_time), np.nan)

##--Other trace gas data: addressing different start/stop times than AIMMS--##
aimms_start = aimms_time.min()
aimms_end = aimms_time.max()

##--Handle black carbon data with different start/stop times than AIMMS--##
BC_time = SP2.data['Time_UTC']

##--Trim CO data if it starts before AIMMS--##
if BC_time.min() < aimms_start:
    mask_start = BC_time >= aimms_start
    BC_time = BC_time[mask_start]
    BC_mass = BC_mass[mask_start]
    
##--Append CO data with NaNs if it ends before AIMMS--##
if BC_time.max() < aimms_end: 
    missing_times = np.arange(BC_time.max()+1, aimms_end +1)
    BC_time = np.concatenate([BC_time, missing_times])
    BC_mass = np.concatenate([BC_mass, [np.nan]*len(missing_times)])

##--Create a DataFrame for BC data and reindex to AIMMS time, setting non-overlapping times to nan--##
BC_df = pd.DataFrame({'Time_UTC': BC_time, 'BC_mass': BC_mass})
BC_aligned = BC_df.set_index('Time_UTC').reindex(aimms_time)
BC_aligned['BC_mass']= BC_aligned['BC_mass'].where(BC_aligned.index.isin(aimms_time), np.nan)
BC_mass_aligned = BC_aligned['BC_mass']

##--Handle CO data with different start/stop times than AIMMS--##
CO_time = CO.data['Time_UTC']

##--Trim CO data if it starts before AIMMS--##
if CO_time.min() < aimms_start:
    mask_start = CO_time >= aimms_start
    CO_time = CO_time[mask_start]
    CO_conc = CO_conc[mask_start]
    
##--Append CO data with NaNs if it ends before AIMMS--##
if CO_time.max() < aimms_end: 
    missing_times = np.arange(CO_time.max()+1, aimms_end +1)
    CO_time = np.concatenate([CO_time, missing_times])
    CO_conc = np.concatenate([CO_conc, [np.nan]*len(missing_times)])

##--Create a DataFrame for CO data and reindex to AIMMS time, setting non-overlapping times to nan--##
CO_df = pd.DataFrame({'Time_UTC': CO_time, 'CO_ppbv': CO_conc})
CO_aligned = CO_df.set_index('Time_UTC').reindex(aimms_time)
CO_aligned['CO_ppbv']= CO_aligned['CO_ppbv'].where(CO_aligned.index.isin(aimms_time), np.nan)
CO_conc_aligned = CO_aligned['CO_ppbv']

##--Handle CO2 data with different start/stop times than AIMMS--##
CO2_time = CO2.data['Time_UTC']

##--Trim CO2 data if it starts before AIMMS--##
if CO2_time.min() < aimms_start:
    mask_start = CO2_time >= aimms_start
    CO2_time = CO2_time[mask_start]
    CO2_conc = CO2_conc[mask_start]
    
##--Append CO2 data with NaNs if it ends before AIMMS--##
if CO2_time.max() < aimms_end: 
    missing_times = np.arange(CO2_time.max()+1, aimms_end +1)
    CO2_time = np.concatenate([CO2_time, missing_times])
    CO2_conc = np.concatenate([CO2_conc, [np.nan]*len(missing_times)])

##--Create a DataFrame for CO2 data and reindex to AIMMS time, setting non-overlapping times to nan--##
CO2_df = pd.DataFrame({'Time_UTC': CO2_time, 'CO2_ppmv': CO2_conc})
CO2_aligned = CO2_df.set_index('Time_UTC').reindex(aimms_time)
CO2_aligned['CO2_ppmv']=CO2_aligned['CO2_ppmv'].where(CO2_aligned.index.isin(aimms_time), np.nan)
CO2_conc_aligned = CO2_aligned['CO2_ppmv']

##--Handle H2O data with different start/stop times than AIMMS--##
H2O_time = H2O.data['Time_UTC']

##--Trim H2O data if it starts before AIMMS--##
if H2O_time.min() < aimms_start:
    mask_start = H2O_time >= aimms_start
    H2O_time = H2O_time[mask_start]
    H2O_conc = H2O_conc[mask_start]
    
##--Append H2O data with NaNs if it ends before AIMMS--##
if H2O_time.max() < aimms_end: 
    missing_times = np.arange(H2O_time.max()+1, aimms_end +1)
    H2O_time = np.concatenate([H2O_time, missing_times])
    H2O_conc = np.concatenate([H2O_conc, [np.nan]*len(missing_times)])

##--Create a DataFrame for H2O data and reindex to AIMMS time, setting non-overlapping times to nan--##
H2O_df = pd.DataFrame({'Time_UTC': H2O_time, 'H2O_ppmv': H2O_conc})
H2O_aligned = H2O_df.set_index('Time_UTC').reindex(aimms_time)
H2O_aligned['H2O_ppmv']=H2O_aligned['H2O_ppmv'].where(H2O_aligned.index.isin(aimms_time), np.nan)
H2O_conc_aligned = H2O_aligned['H2O_ppmv']

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
    

########################
##--Calculate rBC/CO--##
########################

##--Calculate CO enhancement from background as in Kinase et al https://doi.org/10.5194/acp-25-143-2025--##
##--Pull 5th percentile CO data--##
CO_noNaN = CO_conc_aligned.dropna() # Drop NaNs first
CO_5th = np.percentile(CO_noNaN, 5)

##--Calculate BC/(delta)CO--##
BC_CO = (BC_mass_aligned / CO_noNaN) # (ug/m^3)/ppb


###############
##--BINNING--##
###############

##--Creates a Pandas dataframe with all variables--##
df = pd.DataFrame({'PTemp': potential_temp, 'BC_CO': BC_CO, 'CO_conc':CO_conc_aligned, 
                   'CO2_conc': CO2_conc_aligned, 'H2O_conc':H2O_conc_aligned, 'O3_conc':O3_aligned['O3']})

##--Define desired number of bins here--##
num_bins = 50

##--Compute the minimum and maximum altitude, ignoring NaNs--##
min_alt = df['PTemp'].min(skipna=True)
max_alt = df['PTemp'].max(skipna=True)

##--Create bin edges from min_alt to max_alt--##
bin_edges = np.linspace(min_alt, max_alt, num_bins + 1)

##--Pandas 'cut' splits altitude data into specified number of bins--##
df['PTemp_bin'] = pd.cut(df['PTemp'], bins=bin_edges)

##--Group variables into each altitude bin--## 
##--Observed=false shows all bins, even empty ones--##
binned_df = df.groupby('PTemp_bin', observed=False).agg(
    
   ##--Aggregate data--##
    PTemp_center=('PTemp', 'mean'),
    CO_conc_avg=('CO_conc', 'median'),
    CO_conc_min=('CO_conc', 'min'),
    CO_conc_max=('CO_conc', 'max'),
    CO_conc_25th=('CO_conc', lambda x: x.quantile(0.25)),
    CO_conc_75th=('CO_conc', lambda x: x.quantile(0.75)),
    CO2_conc_avg=('CO2_conc', 'median'),
    CO2_conc_min=('CO2_conc', 'min'),
    CO2_conc_max=('CO2_conc', 'max'),
    CO2_conc_25th=('CO2_conc', lambda x: x.quantile(0.25)),
    CO2_conc_75th=('CO2_conc', lambda x: x.quantile(0.75)),    
    H2O_conc_avg=('H2O_conc', 'median'),
    H2O_conc_min=('H2O_conc', 'min'),
    H2O_conc_max=('H2O_conc', 'max'),
    H2O_conc_25th=('H2O_conc', lambda x: x.quantile(0.25)),
    H2O_conc_75th=('H2O_conc', lambda x: x.quantile(0.75)),
    O3_conc_avg=('O3_conc', 'median'),
    O3_conc_min=('O3_conc', 'min'),
    O3_conc_max=('O3_conc', 'max'),
    O3_conc_25th=('O3_conc', lambda x: x.quantile(0.25)),
    O3_conc_75th=('O3_conc', lambda x: x.quantile(0.75)), 
    BC_CO_avg=('BC_CO', 'median'),
    BC_CO_min=('BC_CO', 'min'),
    BC_CO_max=('BC_CO', 'max'),
    BC_CO_25th=('BC_CO', lambda x: x.quantile(0.25)),
    BC_CO_75th=('BC_CO', lambda x: x.quantile(0.75))
    
##--Reset the index so Altitude_bin is just a column--##
).reset_index()


################
##--PLOTTING--##
################

##ADD SECOND Y-AXIS ON THE RIGHT WITH PRESSURE OR POTENTIAL TEMPERATURE?

##--Creates figure with 5 horizontally stacked subplots sharing a y-axis--##
fig, axs = plt.subplots(1, 5, figsize=(15, 6), sharey=True)

##--First subplot: CO--##
##--Averaged data in each bin is plotted against bin center--##
axs[0].plot(binned_df['CO_conc_avg'], binned_df['PTemp_center'], color='crimson', label='CO')
##--Range is given by filling between data minimum and maximum for each bin--##
axs[0].fill_betweenx(binned_df['PTemp_center'], binned_df['CO_conc_min'], 
                     binned_df['CO_conc_max'], color='palevioletred', alpha=0.3)
axs[0].fill_betweenx(binned_df['PTemp_center'], binned_df['CO_conc_25th'], 
                     binned_df['CO_conc_75th'], color='palevioletred', alpha=0.6)
axs[0].set_xlabel('Concentration (ppbv)')
axs[0].set_ylabel('Potential Temperature (K)')
axs[0].set_title('CO')
#axs[0].set_xlim(0, 1500)

##--Add dashed horizontal lines for the polar dome boundaries--##
##--Boundaries are defined from Bozem et al 2019 (ACP)--##
axs[0].axhline(y=285, color='k', linestyle='--', linewidth=1)
axs[0].axhline(y=299, color='k', linestyle='--', linewidth=1)

##--Add text labels on the left-hand side within the plot area--##
##--Compute midpoints for label placement--##
polar_dome_label = 282
marginal_polar_dome_label = 288
x_text = axs[0].get_xlim()[0] + 1  # offset from left edge

axs[0].text(x_text, polar_dome_label, 'Polar Dome',
        rotation=0, fontsize=10, color='k',
        verticalalignment='center', horizontalalignment='left')
axs[0].text(x_text, marginal_polar_dome_label, 'Marginal Dome',
        rotation=0, fontsize=10, color='k',
        verticalalignment='center', horizontalalignment='left')

##--Second subplot: CO2--##
axs[1].plot(binned_df['CO2_conc_avg'], binned_df['PTemp_center'], color='indigo', label='CO2')
axs[1].fill_betweenx(binned_df['PTemp_center'], binned_df['CO2_conc_min'], 
                     binned_df['CO2_conc_max'], color='darkorchid', alpha=0.3)
axs[1].fill_betweenx(binned_df['PTemp_center'], binned_df['CO2_conc_25th'], 
                     binned_df['CO2_conc_75th'], color='darkorchid', alpha=0.45)
axs[1].set_xlabel('Concentration (ppmv)')
axs[1].set_title('CO\u2082')
#axs[1].set_xlim(0, 2000)

axs[1].axhline(y=285, color='k', linestyle='--', linewidth=1)
axs[1].axhline(y=299, color='k', linestyle='--', linewidth=1)

##--Third subplot: O3--##
axs[2].plot(binned_df['O3_conc_avg'], binned_df['PTemp_center'], color='midnightblue', label='O3')
axs[2].fill_betweenx(binned_df['PTemp_center'], binned_df['O3_conc_min'], 
                     binned_df['O3_conc_max'], color='mediumblue', alpha=0.2)
axs[2].fill_betweenx(binned_df['PTemp_center'], binned_df['O3_conc_25th'], 
                     binned_df['O3_conc_75th'], color='mediumblue', alpha=0.3)
axs[2].set_xlabel('Concentration (ppbv)')
axs[2].set_title('O\u2083')
#axs[2].set_xlim(0, 1000)

axs[2].axhline(y=285, color='k', linestyle='--', linewidth=1)
axs[2].axhline(y=299, color='k', linestyle='--', linewidth=1)

##--Fourth subplot: H2O--##
axs[3].plot(binned_df['H2O_conc_avg'], binned_df['PTemp_center'], color='#00555a', label='H2O')
axs[3].fill_betweenx(binned_df['PTemp_center'], binned_df['H2O_conc_min'], 
                     binned_df['H2O_conc_max'], color='#08787f', alpha=0.2)
axs[3].fill_betweenx(binned_df['PTemp_center'], binned_df['H2O_conc_25th'], 
                     binned_df['H2O_conc_75th'], color='#08787f', alpha=0.3)
axs[3].set_xlabel('Concentration (ppmv)')
axs[3].set_title('H\u2082O')
#axs[3].set_xlim(0, 1000)

axs[3].axhline(y=285, color='k', linestyle='--', linewidth=1)
axs[3].axhline(y=299, color='k', linestyle='--', linewidth=1)

##--Fifth subplot: BC/CO enhancement--##
axs[4].plot(binned_df['BC_CO_avg'], binned_df['PTemp_center'], color='darkgreen', label='rBC/CO')
axs[4].fill_betweenx(binned_df['PTemp_center'], binned_df['BC_CO_min'], 
                     binned_df['BC_CO_max'], color='seagreen', alpha=0.2)
axs[4].fill_betweenx(binned_df['PTemp_center'], binned_df['BC_CO_25th'], 
                     binned_df['BC_CO_75th'], color='seagreen', alpha=0.3)
axs[4].set_xlabel('Enhancement ((µg/m\u00b3)/ppmv)')
axs[4].set_title('rBC/CO')
axs[4].set_xlim(-0.01, 0.01)

axs[4].axhline(y=285, color='k', linestyle='--', linewidth=1)
axs[4].axhline(y=299, color='k', linestyle='--', linewidth=1)

##--Use f-string to embed flight # variable in plot title--##
plt.suptitle(f"Vertical Profiles of Trace Gases - {flight.replace('Flight', 'Flight ')}", fontsize=16)

##--Adjusts layout to prevent overlapping--## 
plt.tight_layout(rect=[0, 0, 1, 0.99])

##--Use f-string to save file with flight# appended--##
output_path = f"{output_path}\\{flight}"
plt.savefig(output_path, dpi=300, bbox_inches='tight') 

plt.show()

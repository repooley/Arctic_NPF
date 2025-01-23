# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 12:43:44 2024

@author: repooley
"""
import os
import glob
import icartt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

#########################
##--Open ICARTT Files--##
#########################

##--Set the base directory to project folder--##
directory = r"C:\Users\repooley\REP_PhD\NETCARE2015\data"

##--Select flight (Flight1 thru Flight10)--##
flight = "Flight10" ##--Flight1 AIMMS file currently broken at line 13234--##

##--Define function that creates datasets from filenames--##
def find_files(directory, flight, partial_name):
    ##--flight data are stored in a folder called "raw"--##
    flight_dir = os.path.join(directory, "raw", flight)
    search_pattern = os.path.join(flight_dir, f"*{partial_name}*")
    return sorted(glob.glob(search_pattern))

##--Meterological data from AIMMS monitoring system--##
aimms = icartt.Dataset(find_files(directory, flight, "AIMMS_POLAR6")[0])

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
altitude = aimms.data['Alt']
aimms_time =aimms.data['TimeWave']

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

###############
##--BINNING--##
###############

##--Creates a Pandas dataframe with all variables--##
df = pd.DataFrame({'Altitude': altitude, 'CO_conc':CO_conc_aligned, 
                   'CO2_conc': CO2_conc_aligned, 'H2O_conc':H2O_conc_aligned, 'O3_conc':O3_aligned['O3']})

##--Define desired number of bins here--##
num_bins = 50

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
    
   ##--Aggregate data--##
    Altitude_center=('Altitude', 'mean'),
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
    O3_conc_75th=('O3_conc', lambda x: x.quantile(0.75))
    
##--Reset the index so Altitude_bin is just a column--##
).reset_index()

################
##--PLOTTING--##
################

##ADD SECOND Y-AXIS ON THE RIGHT WITH PRESSURE OR POTENTIAL TEMPERATURE?

##--Creates figure with 4 horizontally stacked subplots sharing a y-axis--##
fig, axs = plt.subplots(1, 4, figsize=(12, 6), sharey=True)

##--First subplot: CO--##
##--Averaged data in each bin is plotted against bin center--##
axs[0].plot(binned_df['CO_conc_avg'], binned_df['Altitude_center'], color='crimson', label='CO')
##--Range is given by filling between data minimum and maximum for each bin--##
axs[0].fill_betweenx(binned_df['Altitude_center'], binned_df['CO_conc_min'], 
                     binned_df['CO_conc_max'], color='palevioletred', alpha=0.3)
axs[0].fill_betweenx(binned_df['Altitude_center'], binned_df['CO_conc_25th'], 
                     binned_df['CO_conc_75th'], color='palevioletred', alpha=0.6)
axs[0].set_xlabel('Concentration (ppbv)')
axs[0].set_ylabel('Altitude (m)')
axs[0].set_title('CO')
#axs[0].set_xlim(0, 1500)

##--Second subplot: CO2--##
axs[1].plot(binned_df['CO2_conc_avg'], binned_df['Altitude_center'], color='indigo', label='CO2')
axs[1].fill_betweenx(binned_df['Altitude_center'], binned_df['CO2_conc_min'], 
                     binned_df['CO2_conc_max'], color='darkorchid', alpha=0.3)
axs[1].fill_betweenx(binned_df['Altitude_center'], binned_df['CO2_conc_25th'], 
                     binned_df['CO2_conc_75th'], color='darkorchid', alpha=0.45)
axs[1].set_xlabel('Concentration (ppmv)')
axs[1].set_title('CO\u2082')
#axs[1].set_xlim(0, 2000)

##--Third subplot: H2O--##
axs[2].plot(binned_df['O3_conc_avg'], binned_df['Altitude_center'], color='midnightblue', label='H2O')
axs[2].fill_betweenx(binned_df['Altitude_center'], binned_df['O3_conc_min'], 
                     binned_df['O3_conc_max'], color='mediumblue', alpha=0.2)
axs[2].fill_betweenx(binned_df['Altitude_center'], binned_df['O3_conc_25th'], 
                     binned_df['O3_conc_75th'], color='mediumblue', alpha=0.3)
axs[2].set_xlabel('Concentration (ppbv)')
axs[2].set_title('O\u2083')
#axs[2].set_xlim(0, 1000)

##--Fourth subplot: H2O--##
axs[3].plot(binned_df['H2O_conc_avg'], binned_df['Altitude_center'], color='#00555a', label='H2O')
axs[3].fill_betweenx(binned_df['Altitude_center'], binned_df['H2O_conc_min'], 
                     binned_df['H2O_conc_max'], color='#08787f', alpha=0.2)
axs[3].fill_betweenx(binned_df['Altitude_center'], binned_df['H2O_conc_25th'], 
                     binned_df['H2O_conc_75th'], color='#08787f', alpha=0.3)
axs[3].set_xlabel('Concentration (ppmv)')
axs[3].set_title('H\u2082O')
#axs[3].set_xlim(0, 1000)

##--Use f-string to embed flight # variable in plot title--##
plt.suptitle(f"Vertical Profiles of Trace Gases - {flight.replace('Flight', 'Flight ')}", fontsize=16)

##--Adjusts layout to prevent overlapping--## 
plt.tight_layout(rect=[0, 0, 1, 0.99])

##--Base output path in directory--##
output_path = r"C:\Users\repooley\REP_PhD\NETCARE2015\data\processed\AltitudeBinnedData\TraceGas"

##--Use f-string to save file with flight# appended--##
output_path = f"{output_path}\\{flight}"
plt.savefig(output_path, dpi=300, bbox_inches='tight') 

plt.show()

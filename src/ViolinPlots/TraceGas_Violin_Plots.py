# -*- coding: utf-8 -*-
"""
Created on Wed May 28 11:38:51 2025

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
directory = r"C:\Users\repooley\REP_PhD\NETCARE2015\data"

##--Select flight (Flight2 thru Flight10)--##
##--FLIGHT1 HAS NO UHSAS FILES--##
flight = "Flight2"

##--Base output path for figures in directory--##
output_path = r"C:\Users\repooley\REP_PhD\NETCARE2015\data\processed\ViolinPlots\TraceGas"

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

##--Trace gases--##
CO = icartt.Dataset(find_files(directory, flight, "CO_POLAR6")[0])
CO2 = icartt.Dataset(find_files(directory, flight, "CO2_POLAR6")[0])

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

##--O3 data--##
##--Put O3 data in list to make concatenation easier--##
O3_starttime = list(O3.data['Start_UTC'])
O3_conc = list(O3.data['O3'])

##--Check for O3_2 data and stich to end of O3 if populated--##
if O3_2 is not None:
    O3_starttime += list(O3_2.data['Start_UTC'])
    O3_conc += list(O3_2.data['O3'])

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
O3_conc_aligned = O3_aligned['O3']

##--CO and CO2--##
CO_conc = CO.data['CO_ppbv']
CO_time = CO.data['Time_UTC']
CO2_conc = CO2.data['CO2_ppmv']
CO2_time = CO2.data['Time_UTC']

CO_df = pd.DataFrame({'time': CO_time, 'conc': CO_conc}).set_index('time')
CO_conc_aligned = CO_df.reindex(aimms_time)['conc']

CO2_df = pd.DataFrame({'time':CO2_time, 'conc': CO2_conc}).set_index('time')
CO2_conc_aligned = CO2_df.reindex(aimms_time)['conc']

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

##--Convert to STP--##

P_STP = 101325  # Pa
T_STP = 273.15  # K

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

##--CO--##

CO_n_3_10 = pd.DataFrame({'CO': CO_conc_aligned, 'Nucleation': nuc_particles,
                                 'LoD': nuc_error_3sigma})
CO_npf = CO_n_3_10['CO'][CO_n_3_10['Nucleation'] > CO_n_3_10['LoD']]
CO_nonpf = CO_n_3_10['CO'][CO_n_3_10['Nucleation'] <= CO_n_3_10['LoD']]
CO_df = {'NPF': CO_npf, 'No NPF': CO_nonpf}

##--CO2--##

CO2_n_3_10 = pd.DataFrame({'CO2': CO2_conc_aligned, 'Nucleation': nuc_particles,
                                 'LoD': nuc_error_3sigma})
CO2_npf = CO2_n_3_10['CO2'][CO2_n_3_10['Nucleation'] > CO2_n_3_10['LoD']]
CO2_nonpf = CO2_n_3_10['CO2'][CO2_n_3_10['Nucleation'] <= CO2_n_3_10['LoD']]
CO2_df = {'NPF': CO2_npf, 'No NPF': CO2_nonpf}

##--O3--##

O3_n_3_10 = pd.DataFrame({'O3': O3_conc_aligned, 'Nucleation': nuc_particles,
                                 'LoD': nuc_error_3sigma})
O3_npf = O3_n_3_10['O3'][O3_n_3_10['Nucleation'] > O3_n_3_10['LoD']]
O3_nonpf = O3_n_3_10['O3'][O3_n_3_10['Nucleation'] <= O3_n_3_10['LoD']]
O3_df = {'NPF': O3_npf, 'No NPF': O3_nonpf}

##--CO/CO2--##

CO_CO2_ratio = CO_conc_aligned / CO2_conc_aligned

CO_CO2_n_3_10 = pd.DataFrame({'CO_CO2': CO_CO2_ratio, 'Nucleation': nuc_particles,
                                 'LoD': nuc_error_3sigma})
CO_CO2_npf = CO_CO2_n_3_10['CO_CO2'][CO_CO2_n_3_10['Nucleation'] > CO_CO2_n_3_10['LoD']]
CO_CO2_nonpf = CO_CO2_n_3_10['CO_CO2'][CO_CO2_n_3_10['Nucleation'] <= CO_CO2_n_3_10['LoD']]
CO_CO2_df = {'NPF': CO_CO2_npf, 'No NPF': CO_CO2_nonpf}

##--CO/rBC--##

# come back to this 

#############
##--Stats--##
#############

##--Counts--##
CO_npf_count = len(CO_npf)
CO_nonpf_count = len(CO_nonpf)
CO2_npf_count = len(CO2_npf)
CO2_nonpf_count = len(CO2_nonpf)
O3_npf_count = len(CO_npf)
O3_nonpf_count = len(O3_nonpf)
CO_CO2_npf_count = len(CO_CO2_npf)
CO_CO2_nonpf_count = len(CO_CO2_nonpf)

##--Statistical signficance for unpaired non-parametric data: Mann-Whitney U test--##
CO_npf_array = CO_npf.dropna().tolist() # data should be in a list or array
CO_nonpf_array = CO_nonpf.dropna().tolist()
U_CO, p_CO = mannwhitneyu(CO_npf_array, CO_nonpf_array)

CO2_npf_array = CO2_npf.dropna().tolist()
CO2_nonpf_array = CO2_nonpf.dropna().tolist()
U_CO2, p_CO2 = mannwhitneyu(CO2_npf_array, CO2_nonpf_array)

O3_npf_array = O3_npf.dropna().tolist()
O3_nonpf_array = O3_nonpf.dropna().tolist()
U_O3, p_O3 = mannwhitneyu(O3_npf_array, O3_nonpf_array)

CO_CO2_npf_array = CO_CO2_npf.dropna().tolist()
CO_CO2_nonpf_array = CO_CO2_nonpf.dropna().tolist()
U_CO_CO2, p_CO_CO2 = mannwhitneyu(CO_CO2_npf_array, CO_CO2_nonpf_array)

################
##--Plotting--##
################

##--Assign correct colors to high or low latitude flights--##
##--Separate flights by latitude--##
high_lat_flights = {'Flight2', 'Flight3', 'Flight4', 'Flight5', 'Flight6', 'Flight7'}

if flight in high_lat_flights: 
    palette = {'NPF':'#219eaf', 'No NPF':'#135d66'}
    palette2 = {'NPF':'#0bafc5', 'No NPF':'#088395'}
    palette3 = {'NPF':'#13adb5', 'No NPF':'#0e8388'}
    palette4 = {'NPF':'#769da6', 'No NPF':'#577d86'}
    palette5 = {'NPF':'#b9e3f7', 'No NPF':'#9ed9f5'} # edit when CO/rBC is done
else: 
    palette = {'NPF':'#be1857', 'No NPF':'#85113d'}
    palette2 = {'NPF':'#bd218d', 'No NPF':'#8c1868'}
    palette3 = {'NPF':'#c51d4d', 'No NPF':'#8f1537'}
    palette4 = {'NPF':'#a31919', 'No NPF':'#6d1111'}
    palette5 = {'NPF':'#d17575', 'No NPF':'#c14545'} # edit when CO/rBC is done

##--CO--##

fig, ax = plt.subplots(figsize = (4,6))
##--Cut=0 disallows interpolation beyond the data extremes. Remove inner box whiskers for clarity--##
CO_plot = sns.violinplot(data=CO_df, palette=palette, ax=ax, cut=0, inner_kws={'whis_width': 0, 'solid_capstyle':'butt'})
ax.set(xlabel='')
ax.set(ylabel='CO (ppmv)')
ax.set(title=f"CO - {flight.replace('Flight', 'Flight ')}")

##--Add text labels with N--##
plt.text(0.25, 0.12, "N={}".format(CO_npf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')
plt.text(0.63, 0.12, "N={}".format(CO_nonpf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')

##--Conditions for adding p values--##
if p_CO >= 0.05:
    plt.text(0.45, 0.85, f"p={p_CO:.4f}", transform=fig.transFigure, fontsize=10, color='orange')
elif 0.05 > p_CO >= 0.0005:
    plt.text(0.45, 0.85, f"p={p_CO:.4f}", transform=fig.transFigure, fontsize=10, color='green')
elif p_CO < 0.0005: 
    plt.text(0.45, 0.85, "p<<0.05", transform=fig.transFigure, fontsize=10, color='green')
 
    
plt.savefig(f"{output_path}\\CO/CO_{flight}", dpi=600)
plt.show()

##--CO2--##

fig, ax = plt.subplots(figsize=(4,6))
CO2_plot = sns.violinplot(data = CO2_df, order=['NPF', 'No NPF'], palette=palette2,
                            ax=ax, cut=0, inner_kws={'whis_width': 0, 'solid_capstyle':'butt'})
ax.set(xlabel='')
ax.set(ylabel='CO2 (ppmv)')
ax.set(title=f"CO2 - {flight.replace('Flight', 'Flight ')}")
plt.text(0.25, 0.12, "N={}".format(CO2_npf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')
plt.text(0.63, 0.12, "N={}".format(CO2_nonpf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')

##--Conditions for adding p values--##
if p_CO2 >= 0.05:
    plt.text(0.45, 0.85, f"p={p_CO2:.4f}", transform=fig.transFigure, fontsize=10, color='orange')
elif 0.05 > p_CO2 >= 0.0005:
    plt.text(0.45, 0.85, f"p={p_CO2:.4f}", transform=fig.transFigure, fontsize=10, color='green')
elif p_CO2 < 0.0005: 
    plt.text(0.45, 0.85, "p<<0.05", transform=fig.transFigure, fontsize=10, color='green')
    
plt.savefig(f"{output_path}\\CO2/CO2_{flight}", dpi=600)
plt.show()

##--O3--##

fig, ax = plt.subplots(figsize=(4,6))
O3_plot = sns.violinplot(data = O3_df, order=['NPF', 'No NPF'], palette=palette3,
                          ax=ax, cut=0, inner_kws={'whis_width': 0, 'solid_capstyle':'butt'})
ax.set(xlabel='')
ax.set(ylabel='O3 (ppbv)')
ax.set(title=f"O3 - {flight.replace('Flight', 'Flight ')}")
plt.text(0.25, 0.12, "N={}".format(O3_npf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')
plt.text(0.63, 0.12, "N={}".format(O3_nonpf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')

##--Conditions for adding p values--##
if p_O3 >= 0.05:
    plt.text(0.45, 0.85, f"p={p_O3:.4f}", transform=fig.transFigure, fontsize=10, color='orange')
elif 0.05 > p_O3 >= 0.0005:
    plt.text(0.45, 0.85, f"p={p_O3:.4f}", transform=fig.transFigure, fontsize=10, color='green')
elif p_O3 < 0.0005: 
    plt.text(0.45, 0.85, "p<<0.05", transform=fig.transFigure, fontsize=10, color='green')
    
    
plt.savefig(f"{output_path}\\O3/O3_{flight}", dpi=600)
plt.show()

##--CO/CO2 ratio--##

fig, ax = plt.subplots(figsize=(4,6))
CO_CO2_plot = sns.violinplot(data = CO_CO2_df, order=['NPF', 'No NPF'], palette=palette4,
                          ax=ax, cut=0, inner_kws={'whis_width': 0, 'solid_capstyle':'butt'})
ax.set(xlabel='')
ax.set(ylabel='CO/CO2 ratio')
ax.set(title=f"CO/CO2 ratio - {flight.replace('Flight', 'Flight ')}")
plt.text(0.25, 0.12, "N={}".format(CO_CO2_npf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')
plt.text(0.63, 0.12, "N={}".format(CO_CO2_nonpf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')

##--Conditions for adding p values--##
if p_CO_CO2 >= 0.05:
    plt.text(0.45, 0.85, f"p={p_CO_CO2:.4f}", transform=fig.transFigure, fontsize=10, color='orange')
elif 0.05 > p_CO_CO2 >= 0.0005:
    plt.text(0.45, 0.85, f"p={p_CO_CO2:.4f}", transform=fig.transFigure, fontsize=10, color='green')
elif p_CO_CO2 < 0.0005: 
    plt.text(0.45, 0.85, "p<<0.05", transform=fig.transFigure, fontsize=10, color='green')

plt.savefig(f"{output_path}\\CO_CO2/CO_CO2_{flight}", dpi=600)
plt.show()
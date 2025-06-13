# -*- coding: utf-8 -*-
"""
Created on Tue May 27 09:37:49 2025

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
flight = "Flight10"

##--Base output path for figures in directory--##
output_path = r"C:\Users\repooley\REP_PhD\NETCARE2015\data\processed\ViolinPlots\Meteorological"

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

##--CPC data--##
CPC10 = icartt.Dataset(find_files(directory, flight, 'CPC3772')[0])
CPC3 = icartt.Dataset(find_files(directory, flight, 'CPC3776')[0])

##--Pull H2O mixing data to calculate RH--##
H2O = icartt.Dataset(find_files(directory, flight, "H2O_POLAR6")[0])

#########################
##--Pull & align data--##
#########################

##--AIMMS Data--##
altitude = aimms.data['Alt'] # in m
latitude = aimms.data['Lat'] # in degrees
temperature = aimms.data['Temp'] + 273.15 # in K
pressure = aimms.data['BP'] # in pa
aimms_time =aimms.data['TimeWave'] # seconds since midnight

##--Ensure H2O data start/stop time is aligned with AIMMS--##
aimms_start = aimms_time.min()
aimms_end = aimms_time.max()

H2O_time = H2O.data['Time_UTC']
H2O_conc = H2O.data['H2O_ppmv']

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
    
#####################
##--Calc humidity--##
#####################

temperature_c = aimms.data['Temp']

##--Convert H2O ppm to RH wrt Water--##

##--Lowe and Ficke (1974) 6th deg polynomial approach--##
##--Sat vap pressure water -50 to 50 C--##
wa0 = 6.107799961
wa1 = 4.436518521E-1
wa2 = 1.428945805E-2
wa3 = 2.650648471E-4
wa4 = 3.031240396E-6
wa5 = 2.034080948E-8
wa6 = 6.136820929E-11

##--Generate empty lists for humididy outputs--##
saturation_humidity_w = []
relative_humidity_w = []

##--Calculate saturation humidity in ppmv and relative humidity--##
for T, P, H2O_ppmv in zip(temperature_c, pressure, H2O_conc_aligned):
    ##--Only calculate within temp range--##
    if -50 <= T < 50:
        ##--saturation vapor pressure using Lowe and Ficke (1974) eqn--##
        e_sw = wa0 + wa1*T + wa2*(T**2)+ wa3*(T**3)+ wa4*(T**4) + wa5*(T**5) + wa6*(T**6) # in mbar 
        ##--Convert from mbar to pa--##
        e_sw_pa = e_sw*100
        ##--Saturation mixing ratio in ppmv--##
        w_s_ppmv = (e_sw_pa / P) * 1e6
        saturation_humidity_w.append(w_s_ppmv)
        ##--Relative humidity--##
        RH = (H2O_ppmv / w_s_ppmv) * 100  # in %
        relative_humidity_w.append(RH)
    else:
        saturation_humidity_w.append(np.nan)  
        relative_humidity_w.append(np.nan)   

##--With respect to ice--##

##--Lowe and Ficke (1974) 6th deg polynomial approach--##
##--Sat vap pressure ice -50 to 0 C--##
ia0 = 6.109177956
ia1 = 5.034698970E-1
ia2 = 1.886013408E-2
ia3 = 4.176223716E-4
ia4 = 5.824720280E-6
ia5 = 4.838803174E-8
ia6 = 1.838826904E-10

##--Generate empty lists for humidity outputs--##
saturation_humidity_i = []
relative_humidity_i = []

##--Calculate saturation humidity wrt ice in ppmv and RH--##
for T, P, H2O_ppmv in zip(temperature_c, pressure, H2O_conc_aligned):
    ##--Only calculate within temp range--##
    if -50 <= T < 0:
        ##--Saturation vapor pressure using Lowe and Ficke (1974) eqn--##
        e_si = ia0 + ia1*T + ia2*(T**2) + ia3*(T**3) + ia4*(T**4) + ia5*(T**5) + ia6*(T**6)  # in mbar
        ##--Convert from mbar to Pa--##
        e_si_pa = e_si * 100
        ##--Saturation mixing ratio in ppbv--##
        e_si_ppmv = (e_si_pa / P) * 1e6
        saturation_humidity_i.append(e_si_ppmv)
        ##--Relative Humidity--##
        RH_i = (H2O_ppmv / e_si_ppmv) * 100  # in %
        relative_humidity_i.append(RH_i)
    else:
        saturation_humidity_i.append(np.nan)  
        relative_humidity_i.append(np.nan)

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

##--Put N(2.5-10) bin center in a df--##
n_3_10_center = pd.DataFrame([6.25]) # Approximate mean of 2.5 and 10

##--Create a dataframe for N 2.5-10--##
n_3_10 = pd.DataFrame({'time': aimms_time, '6': nuc_particles}).set_index('time')

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

##--Temperature and PTemp--##

temp_n_3_10 = pd.DataFrame({'Temp': temperature, 'PTemp': potential_temp, 'Nucleation': n_3_10['6'],
                                 'LoD': nuc_error_3sigma})

temp_npf = temp_n_3_10['Temp'][temp_n_3_10['Nucleation'] > temp_n_3_10['LoD']]
ptemp_npf = temp_n_3_10['PTemp'][temp_n_3_10['Nucleation'] > temp_n_3_10['LoD']]

temp_nonpf = temp_n_3_10['Temp'][temp_n_3_10['Nucleation'] <= temp_n_3_10['LoD']]
ptemp_nonpf = temp_n_3_10['PTemp'][temp_n_3_10['Nucleation'] <= temp_n_3_10['LoD']]

temp_df = {'NPF': temp_npf, 'No NPF': temp_nonpf}
ptemp_df = {'NPF': ptemp_npf, 'No NPF': ptemp_nonpf}

##--Altitude--##

alt_n_3_10 = pd.DataFrame({'Alt': altitude, 'Nucleation': n_3_10['6'],
                                 'LoD': nuc_error_3sigma})
alt_npf = alt_n_3_10['Alt'][alt_n_3_10['Nucleation'] > alt_n_3_10['LoD']]
alt_nonpf = alt_n_3_10['Alt'][alt_n_3_10['Nucleation'] <= alt_n_3_10['LoD']]
alt_df = {'NPF': alt_npf, 'No NPF': alt_nonpf}

##--RH--##
rh_w_n_3_10 = pd.DataFrame({'RH_w': relative_humidity_w, 'Nucleation': n_3_10['6'],
                                 'LoD': nuc_error_3sigma})
rh_i_n_3_10 = pd.DataFrame({'RH_i': relative_humidity_i, 'Nucleation': n_3_10['6'],
                                 'LoD': nuc_error_3sigma})

rh_w_npf = rh_w_n_3_10['RH_w'][rh_w_n_3_10['Nucleation'] > rh_w_n_3_10['LoD']]
rh_i_npf = rh_i_n_3_10['RH_i'][rh_i_n_3_10['Nucleation'] > rh_i_n_3_10['LoD']]

rh_w_nonpf = rh_w_n_3_10['RH_w'][rh_w_n_3_10['Nucleation'] <= rh_w_n_3_10['LoD']]
rh_i_nonpf = rh_i_n_3_10['RH_i'][rh_i_n_3_10['Nucleation'] <= rh_i_n_3_10['LoD']]

rh_w_df = {'NPF': rh_w_npf, 'No NPF': rh_w_nonpf}
rh_i_df = {'NPF': rh_i_npf, 'No NPF': rh_i_nonpf}

#############
##--Stats--##
#############

##--Counts--##
temp_npf_count = len(temp_npf)
temp_nonpf_count = len(temp_nonpf)
ptemp_npf_count = len(ptemp_npf)
ptemp_nonpf_count = len(ptemp_nonpf)
alt_npf_count = len(alt_npf)
alt_nonpf_count = len(alt_nonpf)
rh_w_npf_count = len(rh_w_npf)
rh_w_nonpf_count = len(rh_w_nonpf)
rh_i_npf_count = len(rh_i_npf)
rh_i_nonpf_count = len(rh_i_nonpf)

##--Statistical signficance for unpaired non-parametric data: Mann-Whitney U test--##
temp_npf_array = temp_npf.dropna().tolist() # data should be in a list or array
temp_nonpf_array = temp_nonpf.dropna().tolist()
U_temp, p_temp = mannwhitneyu(temp_npf_array, temp_nonpf_array)

ptemp_npf_array = ptemp_npf.dropna().tolist()
ptemp_nonpf_array = ptemp_nonpf.dropna().tolist()
U_ptemp, p_ptemp = mannwhitneyu(ptemp_npf_array, ptemp_nonpf_array)

alt_npf_array = alt_npf.dropna().tolist()
alt_nonpf_array = alt_nonpf.dropna().tolist()
U_alt, p_alt = mannwhitneyu(alt_npf_array, alt_nonpf_array)

rh_w_npf_array = rh_w_npf.dropna().tolist()
rh_w_nonpf_array = rh_w_nonpf.dropna().tolist()
U_rh_w, p_rh_w = mannwhitneyu(rh_w_npf_array, rh_w_nonpf_array)

rh_i_npf_array = rh_i_npf.dropna().tolist()
rh_i_nonpf_array = rh_i_nonpf.dropna().tolist()
U_rh_i, p_rh_i = mannwhitneyu(rh_i_npf_array, rh_i_nonpf_array)

################
##--Plotting--##
################

##--Assign correct colors to high or low latitude flights--##
##--Separate flights by latitude--##
high_lat_flights = {'Flight2', 'Flight3', 'Flight4', 'Flight5', 'Flight6', 'Flight7'}

if flight in high_lat_flights: 
    palette = {'NPF':'#759fb4', 'No NPF':'#557a96'}
    palette2 = {'NPF':'#acdee5', 'No NPF':'#8bc5d2'}
    palette3 = {'NPF':'#72a4f7', 'No NPF':'#5a95f5'}
    palette4 = {'NPF':'#75ccf5', 'No NPF':'#3fb8f1'}
    palette5 = {'NPF':'#b9e3f7', 'No NPF':'#9ed9f5'}
else: 
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
ax.set(title=f"Temperature - {flight.replace('Flight', 'Flight ')}")

##--Add text labels with N--##
plt.text(0.25, 0.12, "N={}".format(temp_npf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')
plt.text(0.63, 0.12, "N={}".format(temp_nonpf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')

##--Conditions for adding p values--##
if p_temp >= 0.05:
    plt.text(0.45, 0.85, f"p={p_temp:.4f}", transform=fig.transFigure, fontsize=10, color='orange')
elif 0.05 > p_temp >= 0.005:
    plt.text(0.45, 0.85, f"p={p_temp:.4f}", transform=fig.transFigure, fontsize=10, color='green')
elif p_temp < 0.005: 
    plt.text(0.45, 0.85, "p<<0.05", transform=fig.transFigure, fontsize=10, color='green')
 
    
plt.savefig(f"{output_path}\\temp/temp_{flight}", dpi=600)

plt.show()

fig, ax = plt.subplots(figsize=(4,6))
ptemp_plot = sns.violinplot(data = ptemp_df, order=['NPF', 'No NPF'], palette=palette2,
                            ax=ax, cut=0, inner_kws={'whis_width': 0, 'solid_capstyle':'butt'})
ax.set(xlabel='')
ax.set(ylabel='Potential Temperature (K)')
ax.set(title=f"Potential Temperature - {flight.replace('Flight', 'Flight ')}")

##--Add text labels with N--##
plt.text(0.25, 0.12, "N={}".format(ptemp_npf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')
plt.text(0.63, 0.12, "N={}".format(ptemp_nonpf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')

##--Conditions for adding p values--##
if p_ptemp >= 0.05:
    plt.text(0.45, 0.85, f"p={p_ptemp:.4f}", transform=fig.transFigure, fontsize=10, color='orange')
elif 0.05 > p_ptemp >= 0.005:
    plt.text(0.45, 0.85, f"p={p_ptemp:.4f}", transform=fig.transFigure, fontsize=10, color='green')
elif p_ptemp < 0.005: 
    plt.text(0.45, 0.85, "p<<0.05", transform=fig.transFigure, fontsize=10, color='green')
 
    
plt.savefig(f"{output_path}\\ptemp/ptemp_{flight}", dpi=600)

plt.show()

##--ALTITUDE--##

fig, ax = plt.subplots(figsize=(4,6))
alt_plot = sns.violinplot(data = alt_df, order=['NPF', 'No NPF'], palette=palette3,
                          ax=ax, cut=0, inner_kws={'whis_width': 0, 'solid_capstyle':'butt'})
ax.set(xlabel='')
ax.set(ylabel='Altitude AMSL (m)')
ax.set(title=f"Altitude - {flight.replace('Flight', 'Flight ')}")

##--Add text labels with N--##
plt.text(0.25, 0.12, "N={}".format(alt_npf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')
plt.text(0.63, 0.12, "N={}".format(alt_nonpf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')

##--Conditions for adding p values--##
if p_alt >= 0.05:
    plt.text(0.45, 0.85, f"p={p_alt:.4f}", transform=fig.transFigure, fontsize=10, color='orange')
elif 0.05 > p_alt >= 0.005:
    plt.text(0.45, 0.85, f"p={p_alt:.4f}", transform=fig.transFigure, fontsize=10, color='green')
elif p_alt < 0.005: 
    plt.text(0.45, 0.85, "p<<0.05", transform=fig.transFigure, fontsize=10, color='green')
 
    
plt.savefig(f"{output_path}\\altitude/alt_{flight}", dpi=600)

plt.show()

##--RH--##

fig, ax = plt.subplots(figsize=(4,6))
alt_plot = sns.violinplot(data = rh_w_df, order=['NPF', 'No NPF'], palette=palette4,
                          ax=ax, cut=0, inner_kws={'whis_width': 0, 'solid_capstyle':'butt'})
ax.set(xlabel='')
ax.set(ylabel='RH with respect to water (%)')
ax.set(title=f"RH w.r.t. Water - {flight.replace('Flight', 'Flight ')}")

##--Add text labels with N--##
plt.text(0.25, 0.12, "N={}".format(rh_w_npf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')
plt.text(0.63, 0.12, "N={}".format(rh_w_nonpf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')

##--Conditions for adding p values--##
if p_rh_w >= 0.05:
    plt.text(0.45, 0.85, f"p={p_rh_w:.4f}", transform=fig.transFigure, fontsize=10, color='orange')
elif 0.05 > p_rh_w >= 0.005:
    plt.text(0.45, 0.85, f"p={p_rh_w:.4f}", transform=fig.transFigure, fontsize=10, color='green')
elif p_rh_w < 0.005: 
    plt.text(0.45, 0.85, "p<<0.05", transform=fig.transFigure, fontsize=10, color='green')
 
    
plt.savefig(f"{output_path}\\rh_water/rh_w_{flight}", dpi=600)

plt.show()

fig, ax = plt.subplots(figsize=(4,6))
alt_plot = sns.violinplot(data = rh_i_df, order=['NPF', 'No NPF'], palette=palette5, 
                          ax=ax, cut=0, inner_kws={'whis_width': 0, 'solid_capstyle':'butt'})
ax.set(xlabel='')
ax.set(ylabel='RH with respect to ice (%)')
ax.set(title=f"RH w.r.t. Ice - {flight.replace('Flight', 'Flight ')}")

##--Add text labels with N--##
plt.text(0.25, 0.12, "N={}".format(rh_i_npf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')
plt.text(0.63, 0.12, "N={}".format(rh_i_nonpf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')

##--Conditions for adding p values--##
if p_rh_i >= 0.05:
    plt.text(0.45, 0.85, f"p={p_rh_i:.4f}", transform=fig.transFigure, fontsize=10, color='orange')
elif 0.05 > p_rh_i >= 0.005:
    plt.text(0.45, 0.85, f"p={p_rh_i:.4f}", transform=fig.transFigure, fontsize=10, color='green')
elif p_rh_i < 0.005: 
    plt.text(0.45, 0.85, "p<<0.05", transform=fig.transFigure, fontsize=10, color='green')
 
    
plt.savefig(f"{output_path}\\rh_ice/rh_i_{flight}", dpi=600)

plt.show()
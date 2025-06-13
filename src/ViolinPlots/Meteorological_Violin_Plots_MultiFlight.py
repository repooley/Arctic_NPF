# -*- coding: utf-8 -*-
"""
Created on Tue May 27 10:55:11 2025

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
directory = r"C:\Users\repooley\REP_PhD\NETCARE2015\data\raw"

##--Choose which flights to analyze here!--##
##--FLIGHT1 HAS NO USHAS FILE--##
# 'Flight7', 'Flight8',
flights_to_analyze = ["Flight2", "Flight3", "Flight4", "Flight5", "Flight6",  'Flight9', 'Flight10']

##--Base output path for figures in directory--##
output_path = r"C:\Users\repooley\REP_PhD\NETCARE2015\data\processed\ViolinPlots\Meteorological"

#########################
##--Open ICARTT Files--##
#########################

##--Define a function to find all flight data--##
def get_all_flights(directory):
    ##--flights are iteratively named Flight1, Flight2, etc--##
    raw_dir = os.path.join(directory)
    return [flight for flight in os.listdir(raw_dir) if 
            os.path.isdir(os.path.join(raw_dir, flight)) and flight.startswith("Flight")]
 
##--Define a function that creates datasets from filenames--##
def find_files(flight_dir, partial_name):
    search_pattern = os.path.join(flight_dir, f"*{partial_name}*")
    return sorted(glob.glob(search_pattern))

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

###########################
##--Per-flight analysis--##
###########################

##--Separate flights by latitude--##
high_lat_flights = {'Flight2', 'Flight3', 'Flight4', 'Flight5', 'Flight6', 'Flight7'}

##--Store processed data for ALL flights here: --##
temp_highlat = []
ptemp_highlat = []
alt_highlat = []
rh_w_highlat = []
rh_i_highlat = []

temp_lowlat = []
ptemp_lowlat = []
alt_lowlat = []
rh_w_lowlat = []
rh_i_lowlat = []

##--Loop through each flight, pulling and analyzing data--##
for flight in flights_to_analyze:
    ##--Follow which flight is processing--##
    print(f"Processing {flight}...")
    ##--Populate flight_dir established in above function--##
    flight_dir = os.path.join(directory, flight)
    ##--Pull meteorological data from AIMMS monitoring system--##
    aimms_files = find_files(flight_dir, "AIMMS_POLAR6")
    if aimms_files:
        aimms = icartt.Dataset(aimms_files[0])
    else:
        print(f"No AIMMS_POLAR6 file found for {flight}. Skipping...")
        continue  
 
    ##--Pull H2O mixing data to calculate RH--##
    H2O_files = find_files(flight_dir, "H2O_POLAR6")
    if H2O_files: 
        H2O = icartt.Dataset(H2O_files[0])
    else: 
        print(f"No H2O_POLAR6 file found for {flight}. Skipping...")
        continue
 
    ##--Pull CPC files--##
    CPC10_files = find_files(flight_dir, 'CPC3772')
    CPC3_files = find_files(flight_dir, 'CPC3776')
 
    if CPC10_files and CPC3_files:
        ##--Make variables containing all CPC dataset objects--##
        CPC10 = icartt.Dataset(CPC10_files[0])
        CPC3 = icartt.Dataset(CPC3_files[0])
    else:
        print(f"Missing CPC data for {flight}. Skipping...")
        continue
    
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
    
    ##--Create series to add--##
    nucleating_series = n_3_10['6']
    lod_series = nuc_error_3sigma
    
    ##--Make dataframes containing all necessary information per variable to group by--##
    temp_df = pd.DataFrame({'Temp': temperature, 'nucleating': nucleating_series, 'LoD': lod_series})
    ptemp_df = pd.DataFrame({'PTemp': potential_temp, 'nucleating': nucleating_series, 'LoD': lod_series})
    alt_df = pd.DataFrame({'Alt': altitude, 'nucleating': nucleating_series, 'LoD': lod_series})
    rh_w_df = pd.DataFrame({'RH_w': relative_humidity_w, 'nucleating': nucleating_series, 'LoD': lod_series})
    rh_i_df = pd.DataFrame({'RH_i': relative_humidity_w, 'nucleating': nucleating_series, 'LoD': lod_series})
    
    ##--Append to correct regional list--##
    
    if flight in high_lat_flights:
        temp_highlat.append(temp_df)
        ptemp_highlat.append(ptemp_df)
        alt_highlat.append(alt_df)
        rh_w_highlat.append(rh_w_df)
        rh_i_highlat.append(rh_i_df)
        print(f"{flight}: added to HIGH-lat list")
    else:
        temp_lowlat.append(temp_df)
        ptemp_lowlat.append(ptemp_df)
        alt_lowlat.append(alt_df)
        rh_w_lowlat.append(rh_w_df)
        rh_i_lowlat.append(rh_i_df)
        print(f"{flight}: added to LOW-lat list")

##--Concatenate the resulting lists of dataframes into single dataframes--##
temp_highlat = pd.concat(temp_highlat)
ptemp_highlat = pd.concat(ptemp_highlat)
alt_highlat = pd.concat(alt_highlat)
rh_w_highlat = pd.concat(rh_w_highlat)
rh_i_highlat = pd.concat(rh_i_highlat)

temp_lowlat = pd.concat(temp_lowlat)
ptemp_lowlat = pd.concat(ptemp_lowlat)
alt_lowlat = pd.concat(alt_lowlat)
rh_w_lowlat = pd.concat(rh_w_lowlat)
rh_i_lowlat = pd.concat(rh_i_lowlat)

#######################################
##--Filter to NPF and non-NPF times--##
#######################################

##--High lat flights--##
temp_highlat_npf = temp_highlat['Temp'][temp_highlat['nucleating']
                                           > temp_highlat['LoD']]
temp_highlat_nonpf = temp_highlat['Temp'][temp_highlat['nucleating']
                                           <= temp_highlat['LoD']]
ptemp_highlat_npf = ptemp_highlat['PTemp'][ptemp_highlat['nucleating']
                                           > ptemp_highlat['LoD']]
ptemp_highlat_nonpf = ptemp_highlat['PTemp'][ptemp_highlat['nucleating']
                                           <= ptemp_highlat['LoD']]
alt_highlat_npf = alt_highlat['Alt'][alt_highlat['nucleating']
                                           > alt_highlat['LoD']]
alt_highlat_nonpf = alt_highlat['Alt'][alt_highlat['nucleating']
                                           <= alt_highlat['LoD']]
rh_w_highlat_npf = rh_w_highlat['RH_w'][rh_w_highlat['nucleating']
                                           > rh_w_highlat['LoD']]
rh_w_highlat_nonpf = rh_w_highlat['RH_w'][rh_w_highlat['nucleating']
                                           <= rh_w_highlat['LoD']]
rh_i_highlat_npf = rh_i_highlat['RH_i'][rh_i_highlat['nucleating']
                                           > rh_i_highlat['LoD']]
rh_i_highlat_nonpf = rh_i_highlat['RH_i'][rh_i_highlat['nucleating']
                                           <= rh_i_highlat['LoD']]

##--Low lat flights--##
temp_lowlat_npf = temp_lowlat['Temp'][temp_lowlat['nucleating']
                                           > temp_lowlat['LoD']]
temp_lowlat_nonpf = temp_lowlat['Temp'][temp_lowlat['nucleating']
                                           <= temp_lowlat['LoD']]
ptemp_lowlat_npf = ptemp_lowlat['PTemp'][ptemp_lowlat['nucleating']
                                           > ptemp_lowlat['LoD']]
ptemp_lowlat_nonpf = ptemp_lowlat['PTemp'][ptemp_lowlat['nucleating']
                                           <= ptemp_lowlat['LoD']]
alt_lowlat_npf = alt_lowlat['Alt'][alt_lowlat['nucleating']
                                           > alt_lowlat['LoD']]
alt_lowlat_nonpf = alt_lowlat['Alt'][alt_lowlat['nucleating']
                                           <= alt_lowlat['LoD']]
rh_w_lowlat_npf = rh_w_lowlat['RH_w'][rh_w_lowlat['nucleating']
                                           > rh_w_lowlat['LoD']]
rh_w_lowlat_nonpf = rh_w_lowlat['RH_w'][rh_w_lowlat['nucleating']
                                           <= rh_w_lowlat['LoD']]
rh_i_lowlat_npf = rh_i_lowlat['RH_i'][rh_i_lowlat['nucleating']
                                           > rh_i_lowlat['LoD']]
rh_i_lowlat_nonpf = rh_i_lowlat['RH_i'][rh_i_lowlat['nucleating']
                                           <= rh_i_lowlat['LoD']]

##--Final dataframes to feed to the violin plots--##
##--Drop index to prevent reindexing issues--##
temp_sorted = pd.DataFrame({
    'High_NPF': temp_highlat_npf.reset_index(drop=True),
    'Low_NPF': temp_lowlat_npf.reset_index(drop=True),
    'High_NoNPF': temp_highlat_nonpf.reset_index(drop=True),
    'Low_NoNPF': temp_lowlat_nonpf.reset_index(drop=True)
})

ptemp_sorted = pd.DataFrame({
    'High_NPF': ptemp_highlat_npf.reset_index(drop=True),
    'Low_NPF': ptemp_lowlat_npf.reset_index(drop=True),
    'High_NoNPF': ptemp_highlat_nonpf.reset_index(drop=True),
    'Low_NoNPF': ptemp_lowlat_nonpf.reset_index(drop=True)
})


alt_sorted = pd.DataFrame({
    'High_NPF': alt_highlat_npf.reset_index(drop=True),
    'Low_NPF': alt_lowlat_npf.reset_index(drop=True),
    'High_NoNPF': alt_highlat_nonpf.reset_index(drop=True),
    'Low_NoNPF': alt_lowlat_nonpf.reset_index(drop=True)
})

rh_w_sorted = pd.DataFrame({
    'High_NPF': rh_w_highlat_npf.reset_index(drop=True),
    'Low_NPF': rh_w_lowlat_npf.reset_index(drop=True),
    'High_NoNPF': rh_w_highlat_nonpf.reset_index(drop=True),
    'Low_NoNPF': rh_w_lowlat_nonpf.reset_index(drop=True)
})

rh_i_sorted = pd.DataFrame({
    'High_NPF': rh_i_highlat_npf.reset_index(drop=True),
    'Low_NPF': rh_i_lowlat_npf.reset_index(drop=True),
    'High_NoNPF': rh_i_highlat_nonpf.reset_index(drop=True),
    'Low_NoNPF': rh_i_lowlat_nonpf.reset_index(drop=True)
})

#############
##--Stats--##
#############

##--Counts--##
temp_hi_npf_count = len(temp_highlat_npf)
temp_hi_nonpf_count = len(temp_highlat_nonpf)
temp_lo_npf_count = len(temp_lowlat_npf)
temp_lo_nonpf_count = len(temp_lowlat_nonpf)

ptemp_hi_npf_count = len(ptemp_highlat_npf)
ptemp_hi_nonpf_count = len(ptemp_highlat_nonpf)
ptemp_lo_npf_count = len(ptemp_lowlat_npf)
ptemp_lo_nonpf_count = len(ptemp_lowlat_nonpf)

alt_hi_npf_count = len(alt_highlat_npf)
alt_hi_nonpf_count = len(alt_highlat_nonpf)
alt_lo_npf_count = len(alt_lowlat_npf)
alt_lo_nonpf_count = len(alt_lowlat_nonpf)


rh_w_hi_npf_count = len(rh_w_highlat_npf)
rh_w_hi_nonpf_count = len(rh_w_highlat_nonpf)
rh_w_lo_npf_count = len(rh_w_lowlat_npf)
rh_w_lo_nonpf_count = len(rh_w_lowlat_nonpf)

rh_i_hi_npf_count = len(rh_i_highlat_npf)
rh_i_hi_nonpf_count = len(rh_i_highlat_nonpf)
rh_i_lo_npf_count = len(rh_i_lowlat_npf)
rh_i_lo_nonpf_count = len(rh_i_lowlat_nonpf)


##--Statistical signficance for unpaired non-parametric data: Mann-Whitney U test--##
temp_hi_npf_array = temp_highlat_npf.dropna().tolist() # data should be in a list or array
temp_hi_nonpf_array = temp_highlat_nonpf.dropna().tolist()
temp_lo_npf_array = temp_lowlat_npf.dropna().tolist()
temp_lo_nonpf_array = temp_lowlat_nonpf.dropna().tolist()

U_hi_temp, p_hi_temp = mannwhitneyu(temp_hi_npf_array, temp_hi_nonpf_array)
U_lo_temp, p_lo_temp = mannwhitneyu(temp_lo_npf_array, temp_lo_nonpf_array)

ptemp_hi_npf_array = ptemp_highlat_npf.dropna().tolist()
ptemp_hi_nonpf_array = ptemp_highlat_nonpf.dropna().tolist()
ptemp_lo_npf_array = ptemp_lowlat_npf.dropna().tolist()
ptemp_lo_nonpf_array = ptemp_lowlat_nonpf.dropna().tolist()

U_hi_ptemp, p_hi_ptemp = mannwhitneyu(ptemp_hi_npf_array, ptemp_hi_nonpf_array)
U_lo_ptemp, p_lo_ptemp = mannwhitneyu(ptemp_lo_npf_array, ptemp_lo_nonpf_array)

alt_hi_npf_array = alt_highlat_npf.dropna().tolist()
alt_hi_nonpf_array = alt_highlat_nonpf.dropna().tolist()
alt_lo_npf_array = alt_lowlat_npf.dropna().tolist()
alt_lo_nonpf_array = alt_lowlat_nonpf.dropna().tolist()

U_hi_alt, p_hi_alt = mannwhitneyu(alt_hi_npf_array, alt_hi_nonpf_array)
U_lo_alt, p_lo_alt = mannwhitneyu(alt_lo_npf_array, alt_lo_nonpf_array)

rh_w_hi_npf_array = rh_w_highlat_npf.dropna().tolist()
rh_w_hi_nonpf_array = rh_w_highlat_nonpf.dropna().tolist()
rh_w_lo_npf_array = rh_w_lowlat_npf.dropna().tolist()
rh_w_lo_nonpf_array = rh_w_lowlat_nonpf.dropna().tolist()

U_hi_rh_w, p_hi_rh_w = mannwhitneyu(rh_w_hi_npf_array, rh_w_hi_nonpf_array)
U_lo_rh_w, p_lo_rh_w = mannwhitneyu(rh_w_lo_npf_array, rh_w_lo_nonpf_array)

rh_i_hi_npf_array = rh_i_highlat_npf.dropna().tolist()
rh_i_hi_nonpf_array = rh_i_highlat_nonpf.dropna().tolist()
rh_i_lo_npf_array = rh_i_lowlat_npf.dropna().tolist()
rh_i_lo_nonpf_array = rh_i_lowlat_nonpf.dropna().tolist()

U_hi_rh_i, p_hi_rh_i = mannwhitneyu(rh_i_hi_npf_array, rh_i_hi_nonpf_array)
U_lo_rh_i, p_lo_rh_i = mannwhitneyu(rh_i_lo_npf_array, rh_i_lo_nonpf_array)

################
##--Plotting--##
################

##--HIGH LAT: BLUES, LOW LAT: REDS. NPF LIGHTER SHADES--##

##--Palette for temperature plot--##
palette = {'High_NPF':'#002962', 'High_NoNPF': '#00043a', 'Low_NPF': '#a0001c', 'Low_NoNPF':'#800016'}

fig, ax = plt.subplots(figsize = (6,8))
##--Cut=0 disallows interpolation beyond the data extremes--##
temp_plot = sns.violinplot(data=temp_sorted, order = ['High_NPF', 'Low_NPF', 'High_NoNPF', 'Low_NoNPF'], 
                           inner_kws={'whis_width': 0, 'solid_capstyle':'butt'}, palette=palette, ax=ax, cut=0)
ax.set(xlabel='')
ax.set(ylabel='Temperature (K)')
ax.set(title='Temperature')

##--Add text labels with N--##
plt.text(0.17, 0.125, "N={}".format(temp_hi_npf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')
plt.text(0.36, 0.125, "N={}".format(temp_hi_nonpf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')
plt.text(0.56, 0.125, "N={}".format(temp_lo_npf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')
plt.text(0.75, 0.125, "N={}".format(temp_lo_nonpf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')

##--Conditions for adding p values--##
if p_hi_temp >= 0.05:
    plt.text(0.27, 0.85, f"p={p_hi_temp:.4f}", transform=fig.transFigure, fontsize=10, color='orange')
elif 0.05 > p_hi_temp >= 0.005:
    plt.text(0.27, 0.85, f"p={p_hi_temp:.4f}", transform=fig.transFigure, fontsize=10, color='green')
elif p_hi_temp < 0.005: 
    plt.text(0.27, 0.85, "p<<0.05", transform=fig.transFigure, fontsize=10, color='green')
    
if p_lo_temp >= 0.05:
    plt.text(0.65, 0.85, f"p={p_lo_temp:.4f}", transform=fig.transFigure, fontsize=10, color='orange')
elif 0.05 > p_lo_temp >= 0.005:
    plt.text(0.65, 0.85, f"p={p_lo_temp:.4f}", transform=fig.transFigure, fontsize=10, color='green')
elif p_lo_temp < 0.005: 
    plt.text(0.65, 0.85, "p<<0.05", transform=fig.transFigure, fontsize=10, color='green')
    
plt.savefig(f"{output_path}\\temp/temp_MultiFlights", dpi=600)

plt.show()

##--Complementary palette for ptemp--##
palette2 = {'High_NPF':'#407ba7', 'High_NoNPF': '#004e89', 'Low_NPF': '#ff002b', 'Low_NoNPF':'#c00021'}

fig, ax = plt.subplots(figsize=(6,8))
ptemp_plot = sns.violinplot(data = ptemp_sorted, order=['High_NPF', 'High_NoNPF', 'Low_NPF', 'Low_NoNPF'],
                                  inner_kws={'whis_width': 0, 'solid_capstyle':'butt'}, palette=palette2, ax=ax, cut=0)
ax.set(xlabel='')
ax.set(ylabel='Potential Temperature (K)')
ax.set(title="Potential Temperature")

##--Add text labels with N--##
plt.text(0.17, 0.125, "N={}".format(ptemp_hi_npf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')
plt.text(0.36, 0.125, "N={}".format(ptemp_hi_nonpf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')
plt.text(0.56, 0.125, "N={}".format(ptemp_lo_npf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')
plt.text(0.75, 0.125, "N={}".format(ptemp_lo_nonpf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')

##--Conditions for adding p values--##
if p_hi_ptemp >= 0.05:
    plt.text(0.27, 0.85, f"p={p_hi_ptemp:.4f}", transform=fig.transFigure, fontsize=10, color='orange')
elif 0.05 > p_hi_ptemp >= 0.005:
    plt.text(0.27, 0.85, f"p={p_hi_ptemp:.4f}", transform=fig.transFigure, fontsize=10, color='green')
elif p_hi_ptemp < 0.005: 
    plt.text(0.27, 0.85, "p<<0.05", transform=fig.transFigure, fontsize=10, color='green')
    
if p_lo_ptemp >= 0.05:
    plt.text(0.65, 0.85, f"p={p_lo_ptemp:.4f}", transform=fig.transFigure, fontsize=10, color='orange')
elif 0.05 > p_lo_ptemp >= 0.005:
    plt.text(0.65, 0.85, f"p={p_lo_ptemp:.4f}", transform=fig.transFigure, fontsize=10, color='green')
elif p_lo_ptemp < 0.005: 
    plt.text(0.65, 0.85, "p<<0.05", transform=fig.transFigure, fontsize=10, color='green')
    
plt.savefig(f"{output_path}\\ptemp/ptemp_MultiFlights", dpi=600)

plt.show()

##--Palette for altitude--##
palette3 = {'High_NPF':'#72a4f7', 'High_NoNPF': '#5a95f5', 'Low_NPF': '#b11f84', 'Low_NoNPF':'#8c1868'}

fig, ax = plt.subplots(figsize=(6,8))
alt_plot = sns.violinplot(data = alt_sorted, order=['High_NPF', 'High_NoNPF', 'Low_NPF', 'Low_NoNPF'],
                                  inner_kws={'whis_width': 0, 'solid_capstyle':'butt'}, palette=palette3, ax=ax, cut=0)
ax.set(xlabel='')
ax.set(ylabel='Altitude A.M.S.L. (m)')
ax.set(title="Altitude")

##--Add text labels with N--##
plt.text(0.17, 0.125, "N={}".format(alt_hi_npf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')
plt.text(0.36, 0.125, "N={}".format(alt_hi_nonpf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')
plt.text(0.56, 0.125, "N={}".format(alt_lo_npf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')
plt.text(0.75, 0.125, "N={}".format(alt_lo_nonpf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')

##--Conditions for adding p values--##
if p_hi_alt >= 0.05:
    plt.text(0.27, 0.85, f"p={p_hi_alt:.4f}", transform=fig.transFigure, fontsize=10, color='orange')
elif 0.05 > p_hi_alt >= 0.005:
    plt.text(0.27, 0.85, f"p={p_hi_alt:.4f}", transform=fig.transFigure, fontsize=10, color='green')
elif p_hi_alt < 0.005: 
    plt.text(0.27, 0.85, "p<<0.05", transform=fig.transFigure, fontsize=10, color='green')
    
if p_lo_alt >= 0.05:
    plt.text(0.65, 0.85, f"p={p_lo_alt:.4f}", transform=fig.transFigure, fontsize=10, color='orange')
elif 0.05 > p_lo_alt >= 0.005:
    plt.text(0.65, 0.85, f"p={p_lo_alt:.4f}", transform=fig.transFigure, fontsize=10, color='green')
elif p_lo_alt < 0.005: 
    plt.text(0.65, 0.85, "p<<0.05", transform=fig.transFigure, fontsize=10, color='green')
    
plt.savefig(f"{output_path}\\altitude/alt_MultiFlights", dpi=600)

plt.show()

##--Palette for RH--##
palette4 = {'High_NPF':'#75ccf5', 'High_NoNPF': '#3fb8f1', 'Low_NPF': '#fd5f5f', 'Low_NoNPF':'#fd2e2e'}

fig, ax = plt.subplots(figsize=(6,8))
rh_w_plot = sns.violinplot(data = rh_w_sorted, order=['High_NPF', 'High_NoNPF', 'Low_NPF', 'Low_NoNPF'],
                                  inner_kws={'whis_width': 0, 'solid_capstyle':'butt'}, palette=palette4, ax=ax, cut=0)
ax.set(xlabel='')
ax.set(ylabel='Relative Humidity (%)')
ax.set(title="Relative Humidity w.r.t. Water")

##--Add text labels with N--##
plt.text(0.17, 0.125, "N={}".format(rh_w_hi_npf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')
plt.text(0.36, 0.125, "N={}".format(rh_w_hi_nonpf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')
plt.text(0.56, 0.125, "N={}".format(rh_w_lo_npf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')
plt.text(0.75, 0.125, "N={}".format(rh_w_lo_nonpf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')

##--Conditions for adding p values--##
if p_hi_rh_w >= 0.05:
    plt.text(0.27, 0.85, f"p={p_hi_rh_w:.4f}", transform=fig.transFigure, fontsize=10, color='orange')
elif 0.05 > p_hi_rh_w >= 0.005:
    plt.text(0.27, 0.85, f"p={p_hi_rh_w:.4f}", transform=fig.transFigure, fontsize=10, color='green')
elif p_hi_rh_w < 0.005: 
    plt.text(0.27, 0.85, "p<<0.05", transform=fig.transFigure, fontsize=10, color='green')
    
if p_lo_rh_w >= 0.05:
    plt.text(0.65, 0.85, f"p={p_lo_rh_w:.4f}", transform=fig.transFigure, fontsize=10, color='orange')
elif 0.05 > p_lo_rh_w >= 0.005:
    plt.text(0.65, 0.85, f"p={p_lo_rh_w:.4f}", transform=fig.transFigure, fontsize=10, color='green')
elif p_lo_rh_w < 0.005: 
    plt.text(0.65, 0.85, "p<<0.05", transform=fig.transFigure, fontsize=10, color='green')
    
plt.savefig(f"{output_path}\\rh_water/rh_w_MultiFlights", dpi=600)

plt.show()

##--Complementary for RH ice--##
palette5 = {'High_NPF':'#b9e3f7', 'High_NoNPF': '#9ed9f5', 'Low_NPF': '#d17575', 'Low_NoNPF':'#c14545'}

fig, ax = plt.subplots(figsize=(6,8))
rh_w_plot = sns.violinplot(data = rh_w_sorted, order=['High_NPF', 'High_NoNPF', 'Low_NPF', 'Low_NoNPF'],
                                  inner_kws={'whis_width': 0, 'solid_capstyle':'butt'}, palette=palette5, ax=ax, cut=0)
ax.set(xlabel='')
ax.set(ylabel='Relative Humidity (%)')
ax.set(title="Relative Humidity w.r.t. Ice")

##--Add text labels with N--##
plt.text(0.17, 0.125, "N={}".format(rh_i_hi_npf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')
plt.text(0.36, 0.125, "N={}".format(rh_i_hi_nonpf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')
plt.text(0.56, 0.125, "N={}".format(rh_i_lo_npf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')
plt.text(0.75, 0.125, "N={}".format(rh_i_lo_nonpf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')

##--Conditions for adding p values--##
if p_hi_rh_i >= 0.05:
    plt.text(0.27, 0.85, f"p={p_hi_rh_i:.4f}", transform=fig.transFigure, fontsize=10, color='orange')
elif 0.05 > p_hi_rh_i >= 0.005:
    plt.text(0.27, 0.85, f"p={p_hi_rh_i:.4f}", transform=fig.transFigure, fontsize=10, color='green')
elif p_hi_rh_i < 0.005: 
    plt.text(0.27, 0.85, "p<<0.05", transform=fig.transFigure, fontsize=10, color='green')
    
if p_lo_rh_i >= 0.05:
    plt.text(0.65, 0.85, f"p={p_lo_rh_i:.4f}", transform=fig.transFigure, fontsize=10, color='orange')
elif 0.05 > p_lo_rh_i >= 0.005:
    plt.text(0.65, 0.85, f"p={p_lo_rh_i:.4f}", transform=fig.transFigure, fontsize=10, color='green')
elif p_lo_rh_i < 0.005: 
    plt.text(0.65, 0.85, "p<<0.05", transform=fig.transFigure, fontsize=10, color='green')
    
plt.savefig(f"{output_path}\\rh_ice/rh_i_MultiFlights", dpi=600)

plt.show()

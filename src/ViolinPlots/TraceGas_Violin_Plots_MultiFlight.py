# -*- coding: utf-8 -*-
"""
Created on Wed May 28 13:17:09 2025

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
flights_to_analyze = ["Flight2", "Flight3", "Flight4", "Flight5", "Flight6", 'Flight7', 'Flight8', 'Flight9', 'Flight10']

##--Base output path for figures in directory--##
output_path = r"C:\Users\repooley\REP_PhD\NETCARE2015\data\processed\ViolinPlots\TraceGas"

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
CO_highlat = []
CO2_highlat = []
O3_highlat = []
CO_CO2_highlat = []

CO_lowlat = []
CO2_lowlat = []
O3_lowlat = []
CO_CO2_lowlat = []

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
 
    CO_files = find_files(flight_dir, "CO_POLAR6")
    CO2_files = find_files(flight_dir, "CO2_POLAR6")
    
    if CO_files and CO2_files:
        CO = icartt.Dataset(CO_files[0])
        CO2 = icartt.Dataset(CO2_files[0])
    else: 
        print(f"Missing CO or CO2 data for {flight}. Skipping...")
        continue
        
    ##--Flight 2 has multiple ozone files requiring special handling--##
    O3_files = find_files(flight_dir, "O3_")
    
    if len(O3_files) == 0:
        raise FileNotFoundError("No O3 files found.")
        
    elif len(O3_files) == 1 or flight != "Flight2": 
        O3 = icartt.Dataset(O3_files[0])
        O3_2 = None
    ##--Special case for Flight 2--##
    else: 
        O3 = icartt.Dataset(O3_files[0])
        O3_2 = icartt.Dataset(O3_files[1])
 
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
    CPC_df = pd.DataFrame({'Altitude': altitude, 'Latitude': latitude, 
                           'CPC3_conc':CPC3_conc_STP, 'CPC10_conc': CPC10_conc_STP})
    
    ##--Calculate N3-10 particles--##
    nuc_particles = (CPC_df['CPC3_conc'] - CPC_df['CPC10_conc'])
    
    ##--Change calculated particle counts less than zero to NaN--##
    nuc_particles = np.where(nuc_particles >= 0, nuc_particles, np.nan)
   
    #############################
    ##--Propagate uncertainty--##
    #############################
    
    ##--The ICARTT files for CPC instruments say 10% uncertainty of meas value - feels conservative for large counts!--##
    ##--Calculate the 3 sigma uncertainty for nucleating particles--##
    
    T_error = 0.3 # K, constant
    P_error = 100 + 0.0005*(pressure)
    
    ##--Use formula for multiplication/division--##
    greater3nm_error = (CPC3_conc_aligned)*(((P_error)/(pressure))**2 + 
            ((T_error)/(temperature))**2 + ((CPC3_sigma)/(CPC3_conc_aligned)))**(0.5)
    greater10nm_error = (CPC10_conc_aligned)*(((P_error)/(pressure))**2 + 
            ((T_error)/(temperature))**2 + ((CPC10_sigma)/(CPC10_conc_aligned)))**(0.5)
    
    ##--Use add/subtract forumula--##
    nuc_error_3sigma = (((greater3nm_error)**2 + (greater10nm_error)**2)**(0.5))*3
    
    ##--Make dataframes containing all necessary information per variable to group by--##
    CO_df = pd.DataFrame({'CO': CO_conc_aligned, 'nucleating': nuc_particles, 'LoD': nuc_error_3sigma})
    CO2_df = pd.DataFrame({'CO2': CO2_conc_aligned, 'nucleating': nuc_particles, 'LoD': nuc_error_3sigma})
    O3_df = pd.DataFrame({'O3': O3_conc_aligned, 'nucleating': nuc_particles, 'LoD': nuc_error_3sigma})
    
    ##--Calculate CO/CO2--##
    CO_CO2_ratio = CO_conc_aligned / CO2_conc_aligned

    CO_CO2_df = pd.DataFrame({'CO_CO2': CO_CO2_ratio, 'nucleating': nuc_particles,
                                     'LoD': nuc_error_3sigma})

    # add CO/rBC
    
    ##--Append to correct regional list--##
    
    if flight in high_lat_flights:
        CO_highlat.append(CO_df)
        CO2_highlat.append(CO2_df)
        O3_highlat.append(O3_df)
        CO_CO2_highlat.append(CO_CO2_df)
        print(f"{flight}: added to HIGH-lat list")
    else:
        CO_lowlat.append(CO_df)
        CO2_lowlat.append(CO2_df)
        O3_lowlat.append(O3_df)
        CO_CO2_lowlat.append(CO_CO2_df)
        print(f"{flight}: added to LOW-lat list")

##--Concatenate the resulting lists of dataframes into single dataframes--##
CO_highlat = pd.concat(CO_highlat)
CO2_highlat = pd.concat(CO2_highlat)
O3_highlat = pd.concat(O3_highlat)
CO_CO2_highlat = pd.concat(CO_CO2_highlat)

CO_lowlat = pd.concat(CO_lowlat)
CO2_lowlat = pd.concat(CO2_lowlat)
O3_lowlat = pd.concat(O3_lowlat)
CO_CO2_lowlat = pd.concat(CO_CO2_lowlat)

#######################################
##--Filter to NPF and non-NPF times--##
#######################################

##--High lat flights--##
CO_highlat_npf = CO_highlat['CO'][CO_highlat['nucleating']
                                           > CO_highlat['LoD']]
CO_highlat_nonpf = CO_highlat['CO'][CO_highlat['nucleating']
                                           <= CO_highlat['LoD']]
CO2_highlat_npf = CO2_highlat['CO2'][CO2_highlat['nucleating']
                                           > CO2_highlat['LoD']]
CO2_highlat_nonpf = CO2_highlat['CO2'][CO2_highlat['nucleating']
                                           <= CO2_highlat['LoD']]
O3_highlat_npf = O3_highlat['O3'][O3_highlat['nucleating']
                                           > O3_highlat['LoD']]
O3_highlat_nonpf = O3_highlat['O3'][O3_highlat['nucleating']
                                           <= O3_highlat['LoD']]
CO_CO2_highlat_npf = CO_CO2_highlat['CO_CO2'][CO_CO2_highlat['nucleating']
                                           > CO_CO2_highlat['LoD']]
CO_CO2_highlat_nonpf = CO_CO2_highlat['CO_CO2'][CO_CO2_highlat['nucleating']
                                           <= CO_CO2_highlat['LoD']]

##--Low lat flights--##
CO_lowlat_npf = CO_lowlat['CO'][CO_lowlat['nucleating']
                                           > CO_lowlat['LoD']]
CO_lowlat_nonpf = CO_lowlat['CO'][CO_lowlat['nucleating']
                                           <= CO_lowlat['LoD']]
CO2_lowlat_npf = CO2_lowlat['CO2'][CO2_lowlat['nucleating']
                                           > CO2_lowlat['LoD']]
CO2_lowlat_nonpf = CO2_lowlat['CO2'][CO2_lowlat['nucleating']
                                           <= CO2_lowlat['LoD']]
O3_lowlat_npf = O3_lowlat['O3'][O3_lowlat['nucleating']
                                           > O3_lowlat['LoD']]
O3_lowlat_nonpf = O3_lowlat['O3'][O3_lowlat['nucleating']
                                           <= O3_lowlat['LoD']]
CO_CO2_lowlat_npf = CO_CO2_lowlat['CO_CO2'][CO_CO2_lowlat['nucleating']
                                           > CO_CO2_lowlat['LoD']]
CO_CO2_lowlat_nonpf = CO_CO2_lowlat['CO_CO2'][CO_CO2_lowlat['nucleating']
                                           <= CO_CO2_lowlat['LoD']]

##--Final dataframes to feed to the violin plots--##
##--Drop index to prevent reindexing issues--##
CO_sorted = pd.DataFrame({
    'High_NPF': CO_highlat_npf.reset_index(drop=True),
    'Low_NPF': CO_lowlat_npf.reset_index(drop=True),
    'High_NoNPF': CO_highlat_nonpf.reset_index(drop=True),
    'Low_NoNPF': CO_lowlat_nonpf.reset_index(drop=True)
})

CO2_sorted = pd.DataFrame({
    'High_NPF': CO2_highlat_npf.reset_index(drop=True),
    'Low_NPF': CO2_lowlat_npf.reset_index(drop=True),
    'High_NoNPF': CO2_highlat_nonpf.reset_index(drop=True),
    'Low_NoNPF': CO2_lowlat_nonpf.reset_index(drop=True)
})


O3_sorted = pd.DataFrame({
    'High_NPF': O3_highlat_npf.reset_index(drop=True),
    'Low_NPF': O3_lowlat_npf.reset_index(drop=True),
    'High_NoNPF': O3_highlat_nonpf.reset_index(drop=True),
    'Low_NoNPF': O3_lowlat_nonpf.reset_index(drop=True)
})

CO_CO2_sorted = pd.DataFrame({
    'High_NPF': CO_CO2_highlat_npf.reset_index(drop=True),
    'Low_NPF': CO_CO2_lowlat_npf.reset_index(drop=True),
    'High_NoNPF': CO_CO2_highlat_nonpf.reset_index(drop=True),
    'Low_NoNPF': CO_CO2_lowlat_nonpf.reset_index(drop=True)
})

#############
##--Stats--##
#############

##--Counts--##
CO_hi_npf_count = len(CO_highlat_npf)
CO_hi_nonpf_count = len(CO_highlat_nonpf)
CO_lo_npf_count = len(CO_lowlat_npf)
CO_lo_nonpf_count = len(CO_lowlat_nonpf)

CO2_hi_npf_count = len(CO2_highlat_npf)
CO2_hi_nonpf_count = len(CO2_highlat_nonpf)
CO2_lo_npf_count = len(CO2_lowlat_npf)
CO2_lo_nonpf_count = len(CO2_lowlat_nonpf)

O3_hi_npf_count = len(O3_highlat_npf)
O3_hi_nonpf_count = len(O3_highlat_nonpf)
O3_lo_npf_count = len(O3_lowlat_npf)
O3_lo_nonpf_count = len(O3_lowlat_nonpf)

CO_CO2_hi_npf_count = len(CO_CO2_highlat_npf)
CO_CO2_hi_nonpf_count = len(CO_CO2_highlat_nonpf)
CO_CO2_lo_npf_count = len(CO_CO2_lowlat_npf)
CO_CO2_lo_nonpf_count = len(CO_CO2_lowlat_nonpf)

##--Statistical signficance for unpaired non-parametric data: Mann-Whitney U test--##
CO_hi_npf_array = CO_highlat_npf.dropna().tolist() # data should be in a list or array
CO_lo_npf_array = CO_lowlat_npf.dropna().tolist()
CO_hi_nonpf_array = CO_highlat_nonpf.dropna().tolist()
CO_lo_nonpf_array = CO_lowlat_nonpf.dropna().tolist()

U_hi_CO, p_hi_CO = mannwhitneyu(CO_hi_npf_array, CO_hi_nonpf_array)
U_lo_CO, p_lo_CO = mannwhitneyu(CO_lo_npf_array, CO_lo_nonpf_array)

##--Calculate Z-score--##
##--Referenced https://datatab.net/tutorial/mann-whitney-u-test--##
z_hi_CO = (U_hi_CO - CO_hi_npf_count*CO_hi_nonpf_count/2)/((CO_hi_npf_count*
            CO_hi_nonpf_count*(CO_hi_npf_count + CO_hi_nonpf_count + 1)/12)**(1/2))
z_lo_CO = (U_lo_CO - CO_lo_npf_count*CO_lo_nonpf_count/2)/((CO_lo_npf_count*
            CO_lo_nonpf_count*(CO_lo_npf_count + CO_lo_nonpf_count + 1)/12)**(1/2))

##--Take absolute value of Z-score--##
z_hi_CO = abs(z_hi_CO)
z_lo_CO = abs(z_lo_CO)

##--Use Z-score to calculate rank biserial correlation, r--##
r_hi_CO = z_hi_CO/((CO_hi_npf_count + CO_hi_nonpf_count)**(1/2))
r_lo_CO = z_lo_CO/((CO_lo_npf_count + CO_lo_nonpf_count)**(1/2))

CO2_hi_npf_array = CO2_highlat_npf.dropna().tolist()
CO2_lo_npf_array = CO2_lowlat_npf.dropna().tolist()
CO2_hi_nonpf_array = CO2_highlat_nonpf.dropna().tolist()
CO2_lo_nonpf_array = CO2_lowlat_nonpf.dropna().tolist()

U_hi_CO2, p_hi_CO2 = mannwhitneyu(CO2_hi_npf_array, CO2_hi_nonpf_array)
U_lo_CO2, p_lo_CO2 = mannwhitneyu(CO2_lo_npf_array, CO2_lo_nonpf_array)

z_hi_CO2 = (U_hi_CO2 - CO2_hi_npf_count*CO2_hi_nonpf_count/2)/((CO2_hi_npf_count*
            CO2_hi_nonpf_count*(CO2_hi_npf_count + CO2_hi_nonpf_count + 1)/12)**(1/2))
z_lo_CO2 = (U_lo_CO2 - CO2_lo_npf_count*CO2_lo_nonpf_count/2)/((CO2_lo_npf_count*
            CO2_lo_nonpf_count*(CO2_lo_npf_count + CO2_lo_nonpf_count + 1)/12)**(1/2))

z_hi_CO2 = abs(z_hi_CO2)
z_lo_CO2 = abs(z_lo_CO2)

r_hi_CO2 = z_hi_CO2/((CO2_hi_npf_count + CO2_hi_nonpf_count)**(1/2))
r_lo_CO2 = z_lo_CO2/((CO2_lo_npf_count + CO2_lo_nonpf_count)**(1/2))

O3_hi_npf_array = O3_highlat_npf.dropna().tolist()
O3_lo_npf_array = O3_lowlat_npf.dropna().tolist()
O3_hi_nonpf_array = O3_highlat_nonpf.dropna().tolist()
O3_lo_nonpf_array = O3_lowlat_nonpf.dropna().tolist()

U_hi_O3, p_hi_O3 = mannwhitneyu(O3_hi_npf_array, O3_hi_nonpf_array)
U_lo_O3, p_lo_O3 = mannwhitneyu(O3_lo_npf_array, O3_lo_nonpf_array)

z_hi_O3 = (U_hi_O3 - O3_hi_npf_count*O3_hi_nonpf_count/2)/((O3_hi_npf_count*
            O3_hi_nonpf_count*(O3_hi_npf_count + O3_hi_nonpf_count + 1)/12)**(1/2))
z_lo_O3 = (U_lo_O3 - O3_lo_npf_count*O3_lo_nonpf_count/2)/((O3_lo_npf_count*
            O3_lo_nonpf_count*(O3_lo_npf_count + O3_lo_nonpf_count + 1)/12)**(1/2))

z_hi_O3 = abs(z_hi_O3)
z_lo_O3 = abs(z_lo_O3)

r_hi_O3 = z_hi_O3/((O3_hi_npf_count + O3_hi_nonpf_count)**(1/2))
r_lo_O3 = z_lo_O3/((O3_lo_npf_count + O3_lo_nonpf_count)**(1/2))

CO_CO2_hi_npf_array = CO_CO2_highlat_npf.dropna().tolist()
CO_CO2_lo_npf_array = CO_CO2_lowlat_npf.dropna().tolist()
CO_CO2_hi_nonpf_array = CO_CO2_highlat_nonpf.dropna().tolist()
CO_CO2_lo_nonpf_array = CO_CO2_lowlat_nonpf.dropna().tolist()

U_hi_CO_CO2, p_hi_CO_CO2 = mannwhitneyu(CO_CO2_hi_npf_array, CO_CO2_hi_nonpf_array)
U_lo_CO_CO2, p_lo_CO_CO2 = mannwhitneyu(CO_CO2_lo_npf_array, CO_CO2_lo_nonpf_array)

z_hi_CO_CO2 = (U_hi_CO_CO2 - CO_CO2_hi_npf_count*CO_CO2_hi_nonpf_count/2)/((CO_CO2_hi_npf_count*
            CO_CO2_hi_nonpf_count*(CO_CO2_hi_npf_count + CO_CO2_hi_nonpf_count + 1)/12)**(1/2))
z_lo_CO_CO2 = (U_lo_CO_CO2 - CO_CO2_lo_npf_count*CO_CO2_lo_nonpf_count/2)/((CO_CO2_lo_npf_count*
            CO_CO2_lo_nonpf_count*(CO_CO2_lo_npf_count + CO_CO2_lo_nonpf_count + 1)/12)**(1/2))

z_hi_CO_CO2 = abs(z_hi_CO_CO2)
z_lo_CO_CO2 = abs(z_lo_CO_CO2)

r_hi_CO_CO2 = z_hi_CO_CO2/((CO_CO2_hi_npf_count + CO_CO2_hi_nonpf_count)**(1/2))
r_lo_CO_CO2 = z_lo_CO_CO2/((CO_CO2_lo_npf_count + CO_CO2_lo_nonpf_count)**(1/2))

################
##--Plotting--##
################

##--HIGH LAT: BLUES, LOW LAT: REDS. NPF LIGHTER SHADES--##

##--Order of label appearances:--##
group_order = ['NPF', 'No NPF', 'NPF', 'No NPF']

##--Palette for CO plot--##
palette = {'High_NPF':'#219eaf', 'High_NoNPF': '#135d66', 'Low_NPF': '#be1857', 'Low_NoNPF':'#85113d'}

##--Use subplots for breaking y-axis--##
fig, (ax_top, ax_bottom) = plt.subplots(ncols=1, nrows=2, figsize=(6,8), sharex=True, 
                                        height_ratios=[1, 8], gridspec_kw={'hspace':0.08})

##--Cut=0 disallows interpolation beyond the data extremes--##
##--Set inner whisker length to zero for better clarity--##
sns.violinplot(data=CO_sorted, order = ['Low_NPF', 'Low_NoNPF', 'High_NPF', 'High_NoNPF'], 
                                   inner_kws={'whis_width': 0, 'solid_capstyle':'butt'}, palette=palette, ax=ax_top, cut=0)
##--Below the break: copy--##
sns.violinplot(data=CO_sorted, order = ['Low_NPF', 'Low_NoNPF', 'High_NPF', 'High_NoNPF'], 
                                   inner_kws={'whis_width': 0, 'solid_capstyle':'butt'}, palette=palette, ax=ax_bottom, cut=0)

##--Set limits above and below the break--##
ax_top.set_ylim(172, 550) 
ax_bottom.set_ylim(93, 172)

##--Remove duplicated spines--##
sns.despine(ax=ax_bottom, right=False)
sns.despine(ax=ax_top, bottom=True, right=False, top=False)

##--Add diagonal break lines--##

ax = ax_top
ax2 = ax_bottom
##--length of break lines--##
d = .015  
##--Top diagonal--##
ax.plot((-d, +d), (-d, +d), transform=ax_top.transAxes, color='k', clip_on=False)
##--Bottom diagonal--##
##--Bottom break — adjust d to match top angle (scale by inverse of height ratio)--##
d_scaled = d * (1 / 8)
ax2.plot((-d, +d), (1 - d_scaled, 1 + d_scaled), transform=ax_bottom.transAxes, color='k', clip_on=False) 

fig.supylabel('CO (ppmv)', fontsize=12, x=0.01)

##--Add secondary x-axis labels for high and low lat regions--##
fig.supxlabel('65-75\u00b0N', fontsize=12, x=0.32, y=0.045)
plt.text(0.64, 0.045, '>75\u00b0N', transform=fig.transFigure, fontsize=12)

plt.suptitle('CO', fontsize=12, y=0.92)

##--Add x-axis label ticks back in--##
ax_bottom.set_xticks(range(len(group_order)))
ax_bottom.set_xticklabels(group_order)
ax_top.tick_params(axis='x', which='both', labelsize=12, top=False, labeltop=False)

##--Add text labels with N--##
plt.text(0.17, 0.125, "N={}".format(CO_hi_npf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')
plt.text(0.36, 0.125, "N={}".format(CO_hi_nonpf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')
plt.text(0.56, 0.125, "N={}".format(CO_lo_npf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')
plt.text(0.75, 0.125, "N={}".format(CO_lo_nonpf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')

##--Conditions for adding p values--##
if p_lo_CO >= 0.05:
    plt.text(0.17, 0.855, f"p={p_lo_CO:.4f},", transform=fig.transFigure, fontsize=12, color='dimgrey')
elif 0.05 > p_lo_CO >= 0.0005:
    plt.text(0.17, 0.855, f"p={p_lo_CO:.4f},", transform=fig.transFigure, fontsize=12, color='dimgrey')
elif p_lo_CO < 0.0005: 
    plt.text(0.17, 0.855, "p<0.0005,", transform=fig.transFigure, fontsize=12, color='dimgrey')
    
##--Add r value next to p-value--##
plt.text(0.33, 0.855, f"r={r_lo_CO:.3f}", transform=fig.transFigure, fontsize=12, color='dimgrey')

if p_hi_CO >= 0.05:
    plt.text(0.56, 0.855, f"p={p_hi_CO:.4f},", transform=fig.transFigure, fontsize=12, color='dimgrey')
elif 0.05 > p_hi_CO >= 0.0005:
    plt.text(0.56, 0.855, f"p={p_hi_CO:.4f},", transform=fig.transFigure, fontsize=12, color='dimgrey')
elif p_hi_CO < 0.0005: 
    plt.text(0.56, 0.855, "p<0.0005,", transform=fig.transFigure, fontsize=12, color='dimgrey')
    
##--Add r value next to p-value--##
plt.text(0.72, 0.855, f"r={r_hi_CO:.3f}", transform=fig.transFigure, fontsize=12, color='dimgrey')
    
plt.savefig(f"{output_path}\\CO/CO_MultiFlights", dpi=600)

plt.show()

palette2 = {'High_NPF':'#0bafc5', 'High_NoNPF': '#088395', 'Low_NPF': '#bd218d', 'Low_NoNPF':'#8c1868'}

fig, ax = plt.subplots(figsize=(6,8))
CO2_plot = sns.violinplot(data = CO2_sorted, order=['Low_NPF', 'Low_NoNPF', 'High_NPF', 'High_NoNPF'],
                                  inner_kws={'whis_width': 0, 'solid_capstyle':'butt'}, palette=palette2, ax=ax, cut=0)

ax.set_xticks(range(len(group_order)))
ax.set_xticklabels(group_order)

##--Add secondary x-axis labels for high and low lat regions--##
fig.supxlabel('65-75\u00b0N', fontsize=12, x=0.32, y=0.045)
plt.text(0.64, 0.045, '>75\u00b0N', transform=fig.transFigure, fontsize=12)

ax.set(xlabel='')
ax.set(ylabel='CO2 (ppmv)')
ax.set(title="CO2")

##--Add text labels with N--##
plt.text(0.17, 0.125, "N={}".format(CO2_hi_npf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')
plt.text(0.36, 0.125, "N={}".format(CO2_hi_nonpf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')
plt.text(0.56, 0.125, "N={}".format(CO2_lo_npf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')
plt.text(0.75, 0.125, "N={}".format(CO2_lo_nonpf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')

##--Conditions for adding p values--##
if p_lo_CO2 >= 0.05:
    plt.text(0.17, 0.855, f"p={p_lo_CO2:.4f},", transform=fig.transFigure, fontsize=12, color='dimgrey')
elif 0.05 > p_lo_CO2 >= 0.0005:
    plt.text(0.17, 0.855, f"p={p_lo_CO2:.4f},", transform=fig.transFigure, fontsize=12, color='dimgrey')
elif p_lo_CO2 < 0.0005: 
    plt.text(0.17, 0.855, "p<0.0005,", transform=fig.transFigure, fontsize=12, color='dimgrey')
    
##--Add r value next to p-value--##
plt.text(0.33, 0.855, f"r={r_lo_CO2:.3f}", transform=fig.transFigure, fontsize=12, color='dimgrey')

if p_hi_CO2 >= 0.05:
    plt.text(0.56, 0.65, f"p={p_hi_CO2:.4f},", transform=fig.transFigure, fontsize=12, color='dimgrey')
elif 0.05 > p_hi_CO2 >= 0.0005:
    plt.text(0.56, 0.65, f"p={p_hi_CO2:.4f},", transform=fig.transFigure, fontsize=12, color='dimgrey')
elif p_hi_CO2 < 0.0005: 
    plt.text(0.56, 0.65, "p<0.0005,", transform=fig.transFigure, fontsize=12, color='dimgrey')

##--Add r value next to p-value--##
plt.text(0.72, 0.65, f"r={r_hi_CO2:.3f}", transform=fig.transFigure, fontsize=12, color='dimgrey')
    
plt.savefig(f"{output_path}\\CO2/CO2_MultiFlights", dpi=600)

plt.show()

palette3 = {'High_NPF':'#13adb5', 'High_NoNPF': '#0e8388', 'Low_NPF': '#c51d4d', 'Low_NoNPF':'#8f1537'}

fig, ax = plt.subplots(figsize=(6,8))
O3_plot = sns.violinplot(data = O3_sorted, order=['Low_NPF', 'Low_NoNPF', 'High_NPF', 'High_NoNPF'],
                                  inner_kws={'whis_width': 0, 'solid_capstyle':'butt'}, palette=palette3, ax=ax, cut=0)

ax.set_xticks(range(len(group_order)))
ax.set_xticklabels(group_order)

##--Add secondary x-axis labels for high and low lat regions--##
fig.supxlabel('65-75\u00b0N', fontsize=12, x=0.32, y=0.045)
plt.text(0.64, 0.045, '>75\u00b0N', transform=fig.transFigure, fontsize=12)

ax.set(xlabel='')
ax.set(ylabel='O3 (ppbv)')
ax.set(title="O3")

##--Add text labels with N--##
plt.text(0.17, 0.125, "N={}".format(O3_hi_npf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')
plt.text(0.36, 0.125, "N={}".format(O3_hi_nonpf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')
plt.text(0.56, 0.125, "N={}".format(O3_lo_npf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')
plt.text(0.75, 0.125, "N={}".format(O3_lo_nonpf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')

##--Conditions for adding p values--##
if p_lo_O3 >= 0.05:
    plt.text(0.17, 0.855, f"p={p_lo_O3:.4f},", transform=fig.transFigure, fontsize=12, color='dimgrey')
elif 0.05 > p_lo_O3 >= 0.0005:
    plt.text(0.17, 0.855, f"p={p_lo_O3:.4f},", transform=fig.transFigure, fontsize=12, color='dimgrey')
elif p_lo_O3 < 0.0005: 
    plt.text(0.17, 0.855, "p<0.0005,", transform=fig.transFigure, fontsize=12, color='dimgrey')
    
##--Add r value next to p-value--##
plt.text(0.33, 0.855, f"r={r_lo_O3:.3f}", transform=fig.transFigure, fontsize=12, color='dimgrey')

if p_hi_O3 >= 0.05:
    plt.text(0.56, 0.75, f"p={p_hi_O3:.4f},", transform=fig.transFigure, fontsize=12, color='dimgrey')
elif 0.05 > p_hi_O3 >= 0.0005:
    plt.text(0.56, 0.75, f"p={p_hi_O3:.4f},", transform=fig.transFigure, fontsize=12, color='dimgrey')
elif p_hi_O3 < 0.0005: 
    plt.text(0.56, 0.75, "p<0.0005,", transform=fig.transFigure, fontsize=12, color='dimgrey')
    
##--Add r value next to p-value--##
plt.text(0.72, 0.75, f"r={r_hi_O3:.3f}", transform=fig.transFigure, fontsize=12, color='dimgrey')
    
plt.savefig(f"{output_path}\\O3/O3_MultiFlights", dpi=600)

plt.show()

palette4 = {'High_NPF':'#769da6', 'High_NoNPF': '#577d86', 'Low_NPF': '#a31919', 'Low_NoNPF':'#6d1111'}

##--Use subplots for breaking y-axis--##
fig, (ax_top, ax_bottom) = plt.subplots(ncols=1, nrows=2, figsize=(6,8), sharex=True, 
                                        height_ratios=[1, 8], gridspec_kw={'hspace':0.08})

##--Cut=0 disallows interpolation beyond the data extremes--##
##--Set inner whisker length to zero for better clarity--##
sns.violinplot(data=CO_CO2_sorted, order = ['Low_NPF', 'Low_NoNPF', 'High_NPF', 'High_NoNPF'], 
                                   inner_kws={'whis_width': 0, 'solid_capstyle':'butt'}, palette=palette4, ax=ax_top, cut=0)
##--Below the break: copy--##
sns.violinplot(data=CO_CO2_sorted, order = ['Low_NPF', 'Low_NoNPF', 'High_NPF', 'High_NoNPF'], 
                                   inner_kws={'whis_width': 0, 'solid_capstyle':'butt'}, palette=palette4, ax=ax_bottom, cut=0)

##--Set limits above and below the break--##
ax_top.set_ylim(0.452, 1.4) 
ax_bottom.set_ylim(0.23, 0.452)

##--Remove duplicated spines--##
sns.despine(ax=ax_bottom, right=False)
sns.despine(ax=ax_top, bottom=True, right=False, top=False)

##--Add diagonal break lines--##

ax = ax_top
ax2 = ax_bottom
##--length of break lines--##
d = .015  
##--Top diagonal--##
ax.plot((-d, +d), (-d, +d), transform=ax_top.transAxes, color='k', clip_on=False)
##--Bottom diagonal--##
##--Bottom break — adjust d to match top angle (scale by inverse of height ratio)--##
d_scaled = d * (1 / 8)
ax2.plot((-d, +d), (1 - d_scaled, 1 + d_scaled), transform=ax_bottom.transAxes, color='k', clip_on=False) 

fig.supylabel('CO/CO2', fontsize=12, x=0.01)
plt.suptitle('CO/CO2', fontsize=12, y=0.92)

##--Add x-axis label ticks back in--##
ax_bottom.set_xticks(range(len(group_order)))
ax_bottom.set_xticklabels(group_order)

##--Add secondary x-axis labels for high and low lat regions--##
fig.supxlabel('65-75\u00b0N', fontsize=12, x=0.32, y=0.045)
plt.text(0.64, 0.045, '>75\u00b0N', transform=fig.transFigure, fontsize=12)

ax_top.tick_params(axis='x', which='both', labelsize=12, top=False, labeltop=False)

##--Add text labels with N--##
plt.text(0.17, 0.125, "N={}".format(CO_CO2_hi_npf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')
plt.text(0.36, 0.125, "N={}".format(CO_CO2_hi_nonpf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')
plt.text(0.56, 0.125, "N={}".format(CO_CO2_lo_npf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')
plt.text(0.75, 0.125, "N={}".format(CO_CO2_lo_nonpf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')

##--Conditions for adding p values--##
if p_lo_CO_CO2 >= 0.05:
    plt.text(0.17, 0.855, f"p={p_lo_CO_CO2:.4f},", transform=fig.transFigure, fontsize=12, color='dimgrey')
elif 0.05 > p_lo_CO_CO2 >= 0.0005:
    plt.text(0.17, 0.855, f"p={p_lo_CO_CO2:.4f},", transform=fig.transFigure, fontsize=12, color='dimgrey')
elif p_lo_CO_CO2 < 0.0005: 
    plt.text(0.17, 0.855, "p<0.0005,", transform=fig.transFigure, fontsize=12, color='dimgrey')

##--Add r value next to p-value--##
plt.text(0.33, 0.855, f"r={r_lo_CO_CO2:.3f}", transform=fig.transFigure, fontsize=12, color='dimgrey')

if p_hi_CO_CO2 >= 0.05:
    plt.text(0.56, 0.855, f"p={p_hi_CO_CO2:.4f},", transform=fig.transFigure, fontsize=12, color='dimgrey')
elif 0.05 > p_hi_CO_CO2 >= 0.0005:
    plt.text(0.56, 0.855, f"p={p_hi_CO_CO2:.4f},", transform=fig.transFigure, fontsize=12, color='dimgrey')
elif p_hi_CO_CO2 < 0.0005: 
    plt.text(0.56, 0.855, "p<0.0005,", transform=fig.transFigure, fontsize=12, color='dimgrey')
    
##--Add r value next to p-value--##
plt.text(0.72, 0.855, f"r={r_hi_CO_CO2:.3f}", transform=fig.transFigure, fontsize=12, color='dimgrey')
 
plt.savefig(f"{output_path}\\CO_CO2/CO_CO2_MultiFlights", dpi=600)

plt.show()

palette5 = {'High_NPF':'#b9e3f7', 'High_NoNPF': '#9ed9f5', 'Low_NPF': '#d17575', 'Low_NoNPF':'#c14545'}
# edit later for CO/rBC

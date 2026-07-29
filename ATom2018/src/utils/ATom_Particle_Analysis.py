# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 11:48:31 2026

@author: repooley
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path

###########################
##--Establish directory--##
###########################

##--Path to this script--##
script_path = Path(__file__).resolve()

##--Path to the root which is 3 levels up in the directory--##
root = script_path.parents[3]

##--Path to raw ATom data--##
directory = root / "ATom2018" / "data" / "raw"

##--Find the files for every flight called--##
def find_files(directory, flight, partial_name):
    search_pattern = os.path.join(directory, flight, f"*{partial_name}*")
    return sorted(glob.glob(search_pattern))

#######################
##--Calculate PTemp--##
#######################

##--Constants--##
p_0 = 1E5 # Reference pressure in Pa (1000 hPa)
k = 0.286 # Poisson constant for dry air

##--Calculation as a function--##
def calc_ptemp(temperature, pressure):
    
    potential_temp = []
    
    for T, P in zip(temperature, pressure):
        p_t = T*(p_0/P)**k
        potential_temp.append(p_t)
        
    return potential_temp


def calc_ATom_particle_bins(data):
    
    #########################
    ##--Open ICARTT Files--##
    #########################

    #################
    ##--Pull data--##
    #################

    ATom_latitude = data.data['G_LAT'] # deg
    ATom_longitude = data.data['G_LONG'] # deg
    ATom_altitude = data.data['G_ALT'] # m
    ATom_temperature = data.data['T'] # in K
    ATom_pressure = data.data['P'] * 100 # in Pa
    ATom_RH = data.data['Relative_Humidity'] # percent
    ATom_time =data.data['UTC_Start'] # seconds since midnight UTC
    ATom_nucleating = data.data['N_nucl_AMP'] # num/cm^3 STP (2.7-12 nm)
    ATom_aitken = data.data['N_aitken_AMP'] # num/cm^3 STP (12-60 nm)
    ATom_accumulation = data.data['N_accum_AMP'] # num/cm^3 STP (60 nm - 0.5 um)
    ATom_coarse = data.data['N_coarse_AMP'] # num/cm^3 STP (0.5 um - 4.8 um)
    ATom_rBC = data.data['BC_mass_90_550_nm'] # ng/m^3 STP
    ATom_rBC[ATom_rBC < 0] = np.nan
    ATom_O3 = data.data['O3_UCATS'] #ppb
    
    ##--First convert to a series for calc--##
    ATom_nucleating_series = pd.Series(ATom_nucleating)
    
    ##--Calculate potential temperature using above function--##
    ATom_potential_temp = calc_ptemp(ATom_temperature, ATom_pressure)
        
    ##--Convert ptemp to np array--##
    ATom_potential_temp = np.array(ATom_potential_temp)
            
    ##--Append Nucleating and Aitken lists with dataframes--##
    ##--Convert nucleating data to dataframe--##
    ATom_nucleating_df = pd.DataFrame({'nucleating': ATom_nucleating_series, 
                                  'latitude': ATom_latitude, 
                                  'PTemp': ATom_potential_temp, 
                                  'time': ATom_time}).set_index(ATom_time)
    
    ##--Convert aitken mode data to dataframe--##
    ATom_aitken_df = pd.DataFrame({'aitken': ATom_aitken,
                              'latitude': ATom_latitude, 
                              'PTemp': ATom_potential_temp, 
                              'time': ATom_time}).set_index(ATom_time)
    
    ATom_accum_df = pd.DataFrame({'accumulation': ATom_accumulation,
                              'latitude': ATom_latitude, 
                              'PTemp': ATom_potential_temp, 
                              'time': ATom_time}).set_index(ATom_time)
    
    ATom_coarse_df = pd.DataFrame({'coarse': ATom_coarse,
                              'latitude': ATom_latitude, 
                              'PTemp': ATom_potential_temp, 
                              'time': ATom_time}).set_index(ATom_time)
    
    ##--Make separate dfs for particle data taken OUT of STP--##
    ##--Condensation and coagulation sink calculations require ambient data--##
    ##--Specify the standard conditions--##
    P_STP = 101325 # Pa
    T_STP = 273.15 # K

    ##--The conversion factor out of STP--##
    STP_conversion = (ATom_pressure / P_STP) * (T_STP / ATom_temperature)

    ATom_aitken_ambient = data.data['N_aitken_AMP'] * STP_conversion
    ATom_accumulation_ambient = data.data['N_accum_AMP'] * STP_conversion
    ATom_coarse_ambient = data.data['N_coarse_AMP'] * STP_conversion
    
    #############################
    ##--Calculate uncertainty--##
    #############################
    
    ##--Reference Brock et al 2019--##
    nucleating_random = 0.13 # +- percent value
    nucleating_random = nucleating_random * ATom_nucleating_df['nucleating'] #series
    nucleating_systematic = 0.20
    nucleating_systematic = nucleating_systematic * ATom_nucleating_df['nucleating']
    aitken_random = 0.06
    aitken_random = aitken_random * ATom_aitken_df['aitken']
    aitken_systematic = 0.10
    aitken_systematic = aitken_systematic * ATom_aitken_df['aitken']
    
    ##--Combine random and systematic uncertainties in quadrature--##
    ##--3 sigma--##
    nuc_uncertainty = 3* np.sqrt((nucleating_random)**2 + (nucleating_systematic)**2)
    aitken_uncertainty = 3* np.sqrt((aitken_random)**2 + (aitken_systematic)**2)
    
    ##--Append dfs with uncertainties--##
    ATom_nucleating_df['nuc_uncertainty'] = nuc_uncertainty
    ATom_aitken_df['aitken_uncertainty'] = aitken_uncertainty
        
    ###########################
    ##--Wrangle binned data--##
    ###########################
    
    ##--Place all binned non-STP data in a single df--##
    ATom_all_bins_aligned = pd.concat([pd.DataFrame({'36':ATom_aitken_ambient, '280':ATom_accumulation_ambient, 
                                        '2650':ATom_coarse_ambient}, index=ATom_time)], axis=1)
    
    ##--Sum across all bins to get total particle count--##
    ATom_total_particle_count = ATom_all_bins_aligned.sum(axis=1, numeric_only=True) 
    
    ##--Create a dictionary to store each column as a separate dataframe, col names are keys--##
    ATom_diameters = {col: pd.DataFrame({col: ATom_all_bins_aligned[col]}) for col in ATom_all_bins_aligned.columns}
    
    ##--Create separate conditions df and append existing list--##
    ATom_conditions_df = pd.DataFrame({'temperature': ATom_temperature, 
                    'pressure': ATom_pressure, 'PTemp': ATom_potential_temp,
                    'RH': ATom_RH, 'rBC': ATom_rBC, 'O3': ATom_O3,
                    'latitude': ATom_latitude, 'longitude': ATom_longitude,
                    'altitude': ATom_altitude}, index=ATom_time)
    
    return {"ATom_nucleating": ATom_nucleating_df, "ATom_aitken": ATom_aitken_df, 
            "ATom_accumulation": ATom_accum_df, "ATom_coarse": ATom_coarse_df,
            "ATom_total_particle_count": ATom_total_particle_count, 
            "ATom_diameters": ATom_diameters,
            "ATom_conditions": ATom_conditions_df}
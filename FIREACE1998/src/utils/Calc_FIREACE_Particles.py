# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 10:23:14 2026

@author: repooley
"""

import pandas as pd
import numpy as np
from pathlib import Path
import icartt

##--Establish root directory relative to this script--##
##--Path to this script--##
script_path = Path(__file__).resolve()

##--Path to the project root which is 3 levels up in the directory--##
root = script_path.parents[3]

##--Pull in PCASP bin info--##
##--Path to csv file containing PCASP bin info--##
PCASP_bins_path = (root / "FIREACE1998" / "data" / "raw" / "FIREACE1998_PCASP_bins.csv") 

PCASP_bins = pd.read_csv(PCASP_bins_path)

##--15 total bins--##
PCASP_bin_num = [f'bin_{i}' for i in range(1, 16)]

##--Information for bins--##
PCASP_bin_center = PCASP_bins['bin_avg']
PCASP_lower_bound = PCASP_bins['lower_bound']
PCASP_upper_bound = PCASP_bins['upper_bound']

##--Path to raw NETCARE data--##
directory = root / "NETCARE2015" / "data" / "raw"

##--Pull R1 data for the two CPC instruments - zeroes not yet filtered out--##
CPC3_R1 = icartt.Dataset(directory / "CPC_R1" / "CPC3776_Polar6_20150408_R1_L2.ict")
CPC10_R1 = icartt.Dataset(directory / "CPC_R1" / "CPC3772_Polar6_20150408_R1_L2.ict")

#######################
##--CPC uncertainty--##
#######################

##--Pull CPC data from R1 data--##
CPC3_R1_conc = CPC3_R1.data['conc']
CPC10_R1_conc = CPC10_R1.data['conc']

##--Isolate zero periods using a cutoff of 10 counts--##
##--Real data periods are all greater than 10 counts, zeros all less than--##
##--Numpy doesn't recognize -9999 as NaN, tell it to ignore these values--##
CPC3_zeros_c = CPC3_R1_conc[(CPC3_R1_conc < 10) & (CPC3_R1_conc != -9999)]
CPC10_zeros_c = CPC10_R1_conc[(CPC10_R1_conc < 10) & (CPC10_R1_conc != -99999)]

##--Calculate standard deviation of zeros--##
##--3 sigma!--##
CPC3_sigma = 3* np.std(CPC3_zeros_c, ddof=1)  # Use ddof=1 for sample standard deviation
CPC10_sigma = 3* np.std(CPC10_zeros_c, ddof=1)

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

##--Calculate the PNSD for FIRE-ACE campaign in reusable function--##
def calc_FIREACE_particles(files):

    ##--The averaged data is always the second file--##
    if files:
        FIREACE_data = pd.read_csv(files[1])
        ##--1 hz is always first--##
        FIREACE_nonav = pd.read_csv(files[0])
    
    ##--Pull data variables from averaged file--##
    FIREACE_time = FIREACE_data['Time'] # HHMMSS UTC time
    FIREACE_pressure = FIREACE_data['Pressure'] * 100 # in Pa
    FIREACE_temperature = FIREACE_data['Temperature'] + 273.15 # in K
    FIREACE_altitude = FIREACE_data['Altitude'] # in m (agl?)
    FIREACE_latitude = FIREACE_data['Latitude'] # degrees
    FIREACE_RH = FIREACE_data['RH'] # % wrt water
    
    ##--Also pull for nonaveraged file--##
    FIREACE_time_nonav = FIREACE_nonav['Time'] # HHMMSS UTC
    FIREACE_pressure_nonav = FIREACE_nonav['Pressure'] * 100 # in Pa
    FIREACE_temperature_nonav = FIREACE_nonav['Temperature'] + 273.15 # in K
    FIREACE_altitude_nonav = FIREACE_nonav['Altitude'] # in m
    FIREACE_latitude_nonav = FIREACE_nonav['Latitude'] # degrees
    FIREACE_longitude_nonav = FIREACE_nonav['Longitude'] # degrees
    FIREACE_RH_nonav = FIREACE_nonav['RH'] # degrees
    
    ##--No rBC data for FIRE-ACE--##
    
    ##--Constrain latitude to the Arctic region--##
    FIREACE_latitude = FIREACE_latitude.where(FIREACE_latitude >= 66.5, np.nan)
    
    ##--Particle data, 3 and 10 nm cutoffs, respectively--##
    FIREACE_CPC3_data = FIREACE_data['CN3025'] # Is there a corrected version?
    FIREACE_CPC10_data = FIREACE_data['CN7610']
    
    ##--Repeat for non-averaged data--##
    FIREACE_CPC3_nonav = FIREACE_nonav['CN3025_corrected'] # Uncorrected data has a flow issue
    FIREACE_CPC10_nonav = FIREACE_nonav['CN7610']
    
    ##--PCASP data--##
    FIREACE_PCASP_data = FIREACE_data.iloc[:, 14:29] 
    
    ##--Add time to PCASP_data--##
    FIREACE_PCASP_data.insert(0, 'Time', FIREACE_data['Time'])
    
    ##--Set time as the index for later alignment--##
    FIREACE_PCASP_data = FIREACE_PCASP_data.set_index('Time')
    
    ##--Create new column names by rounding the bin center values to the nearest integer--##
    PCASP_new_col_names = PCASP_bin_center.round().astype(int).tolist()
    
    ##--Rename the PCASP_bins df columns to bin average values--##
    FIREACE_PCASP_data.columns = PCASP_new_col_names
    
    ######################
    ##--Calc N(2.5-10)--##
    ######################
    
    ##--Convert to STP!--##
    P_STP = 101325  # Pa
    T_STP = 273.15  # K
    
    ##--Create empty list for CPC3 particles--##
    FIREACE_CPC3_conc_STP_nonav = []
    
    ##--Use the NON-AVERAGED data for first calculation--##
    for CPC3, T, P in zip(FIREACE_CPC3_nonav, FIREACE_temperature_nonav, 
    FIREACE_pressure_nonav):
        if np.isnan(CPC3) or np.isnan(T) or np.isnan(P):
            ##--Append with NaN if any input is NaN--##
            FIREACE_CPC3_conc_STP_nonav.append(np.nan)
        else:
            ##--Perform conversion if all inputs are valid--##
            CPC3_conversion = CPC3 * (P_STP / P) * (T / T_STP)
            FIREACE_CPC3_conc_STP_nonav.append(CPC3_conversion)
        
    ##--Create empty list for CPC10 particles--##
    FIREACE_CPC10_conc_STP_nonav = []
    
    for CPC10, T, P in zip(FIREACE_CPC10_nonav, FIREACE_temperature_nonav, 
    FIREACE_pressure_nonav):
        if np.isnan(CPC10) or np.isnan(T) or np.isnan(P):
            ##--Append with NaN if any input is NaN--##
            FIREACE_CPC10_conc_STP_nonav.append(np.nan)
        else:
            ##--Perform conversion if all inputs are valid--##
            CPC10_conversion = CPC10 * (P_STP / P) * (T / T_STP)
            FIREACE_CPC10_conc_STP_nonav.append(CPC10_conversion)
    
    ##--Creates a Pandas dataframe for particle data--##
    FIREACE_particle_df_nonav = pd.DataFrame({'Altitude': FIREACE_altitude_nonav, 
                                       'Latitude': FIREACE_latitude_nonav,
                                       'CPC3_conc':FIREACE_CPC3_conc_STP_nonav, 
                                       'CPC10_conc': FIREACE_CPC10_conc_STP_nonav})
    
    ##--Calculate N3-10 particles--##
    FIREACE_nuc_particles_nonav = (FIREACE_particle_df_nonav['CPC3_conc'] - 
                            FIREACE_particle_df_nonav['CPC10_conc'])
    
    ##--Change calculated particle counts less than zero to NaN--##
    FIREACE_nuc_particles_nonav = np.where(FIREACE_nuc_particles_nonav >= 0, 
                                    FIREACE_nuc_particles_nonav, np.nan)
    
    ##--Add nucleating particles to df--##
    FIREACE_particle_df_nonav['n_3_10'] = FIREACE_nuc_particles_nonav
    
    ##--Repeat calculation for AVERAGED CPC data--##
    ##--Create empty list for CPC3 particles--##
    FIREACE_CPC3_conc_STP = []
    
    ##--Use the NON-AVERAGED data--##
    for CPC3, T, P in zip(FIREACE_CPC3_data, FIREACE_temperature, FIREACE_pressure):
        if np.isnan(CPC3) or np.isnan(T) or np.isnan(P):
            ##--Append with NaN if any input is NaN--##
            FIREACE_CPC3_conc_STP.append(np.nan)
        else:
            ##--Perform conversion if all inputs are valid--##
            CPC3_conversion = CPC3 * (P_STP / P) * (T / T_STP)
            FIREACE_CPC3_conc_STP.append(CPC3_conversion)
        
    ##--Create empty list for CPC10 particles--##
    FIREACE_CPC10_conc_STP = []
    
    for CPC10, T, P in zip(FIREACE_CPC10_data, FIREACE_temperature, FIREACE_pressure):
        if np.isnan(CPC10) or np.isnan(T) or np.isnan(P):
            ##--Append with NaN if any input is NaN--##
            FIREACE_CPC10_conc_STP.append(np.nan)
        else:
            ##--Perform conversion if all inputs are valid--##
            CPC10_conversion = CPC10 * (P_STP / P) * (T / T_STP)
            FIREACE_CPC10_conc_STP.append(CPC10_conversion)
    
    ##--Creates a Pandas dataframe for AVERAGED particle data--##
    FIREACE_particle_df = pd.DataFrame({'Altitude': FIREACE_altitude, 
                                       'Latitude': FIREACE_latitude,
                                       'CPC3_conc':FIREACE_CPC3_conc_STP, 
                                       'CPC10_conc': FIREACE_CPC10_conc_STP})
    
    ##--Calculate AVERAGED N3-10 particles--##
    FIREACE_nuc_particles = (FIREACE_particle_df['CPC3_conc'] - 
                            FIREACE_particle_df['CPC10_conc'])
    
    ##--Change calculated particle counts less than zero to NaN--##
    FIREACE_nuc_particles = np.where(FIREACE_nuc_particles >= 0, 
                                    FIREACE_nuc_particles, np.nan)
    
    ##--Append nucleating particles to averaged df--##
    FIREACE_particle_df["n_3_10"] = FIREACE_nuc_particles
    
    ##--Make separate dfs for ambient non-averaged data--##
    FIREACE_particle_df_ambient = pd.DataFrame({'Altitude': FIREACE_altitude,
                                                'Latitude': FIREACE_latitude,
                                                'CPC3_conc': FIREACE_CPC3_data,
                                                'CPC10_conc': FIREACE_CPC10_data})
    
    ##--Calculate N3-10 particles at ambient conditions--##
    FIREACE_nuc_particles_ambient = (FIREACE_particle_df_ambient['CPC3_conc'] - 
                                     FIREACE_particle_df_ambient['CPC10_conc'])
    
    ##--Change calculated counts less than zero to NaN--##
    FIREACE_nuc_particles_ambient = np.where(FIREACE_nuc_particles_ambient >= 0,
                                             FIREACE_nuc_particles_ambient, np.nan)
    
    ##--Add ambient nucleating data back to the df--##
    FIREACE_particle_df_ambient['n_3_10'] = FIREACE_nuc_particles_ambient 
    
    ############################
    ##--Normalize PCASP data--##
    ############################
    
    ##--Calculate dlogDp for each bin in numpy array--##
    dlogDp = np.log(PCASP_upper_bound.values) - np.log(PCASP_lower_bound.values)
    
    ##--Normalize counts by dividing by dlogDp across all rows--##
    FIREACE_PCASP_dNdlogDp = FIREACE_PCASP_data.divide(dlogDp, axis=1)
    
    ##--Create empty list for PCASP particles--##
    FIREACE_PCASP_STP = []
    
    for PCASP, T, P in zip(FIREACE_PCASP_dNdlogDp.values, 
    FIREACE_data['Temperature']+273.15, FIREACE_data['Pressure']*100):
        if np.isnan(T) or np.isnan(P):
            ##--Append with NaN if any input is NaN--##
            FIREACE_PCASP_STP.append([np.nan]*len(PCASP))
        else:
            ##--Perform conversion if all inputs are valid--##
            corrected_PCASP = PCASP * (P_STP / P) * (T / T_STP)
            FIREACE_PCASP_STP.append(corrected_PCASP)
    
    ##--Convert back to DataFrame with same columns and index--##
    FIREACE_PCASP_STP = pd.DataFrame(FIREACE_PCASP_STP, 
        columns=FIREACE_PCASP_dNdlogDp.columns, index=FIREACE_particle_df.index)
    
    ##--Also compute STP for non-normalized PCASP data to recompute total sum--##
    FIREACE_PCASP_STP_nonnormal = []
    
    for PCASP, T, P in zip(FIREACE_PCASP_data.values,
    FIREACE_data['Temperature']+273.15, FIREACE_data['Pressure']*100):
        if np.isnan(T) or np.isnan(P):
            ##--Append with NaN if any input is NaN--##
            FIREACE_PCASP_STP_nonnormal.append([np.nan]*len(PCASP))
        else:
            ##--Perform conversion if all inputs are valid--##
            corrected_PCASP = PCASP * (P_STP / P) * (T / T_STP)
            FIREACE_PCASP_STP_nonnormal.append(corrected_PCASP)
            
    ##--Convert back to DataFrame with the same columns and index--##
    FIREACE_PCASP_STP_nonnormal = pd.DataFrame(FIREACE_PCASP_STP_nonnormal,
        columns=FIREACE_PCASP_data.columns, index=FIREACE_particle_df.index)
    

    ##--Add PCASP data to the dataframe--##
    FIREACE_particle_df = pd.concat([FIREACE_particle_df, FIREACE_PCASP_STP], axis=1)
    
    ##--Calculate the PCASP Poisson counting uncertainty--##
    FIREACE_PCASP_sqrt = np.sqrt(FIREACE_PCASP_STP_nonnormal)
    
    ##--Sum all the PCASP uncertainties--##
    FIREACE_PCASP_error = FIREACE_PCASP_sqrt.sum(axis=1)
    

    ##--Add PCASP total counts to the dataframe--##
    FIREACE_particle_df['PCTcon'] = FIREACE_PCASP_error
    
    ##--Add PCASP uncertainty to the dataframe--##
    FIREACE_particle_df['PCASP_error'] = FIREACE_PCASP_error
    
    ##--Also do PCASP data at ambient conditions--##
    FIREACE_PCASP = pd.DataFrame(FIREACE_PCASP_dNdlogDp.values,
                                 columns=FIREACE_PCASP_dNdlogDp.columns, 
                                 index=FIREACE_particle_df_ambient.index)
    
    ##--Add PCASP data to the ambient df--##
    FIREACE_particle_df_ambient = pd.concat([FIREACE_particle_df_ambient, FIREACE_PCASP], axis=1)
    
    ##--Calculate ambient total PCASP particle concentration (non-normalized)--##
    FIREACE_PCASP_sum_ambient = FIREACE_PCASP_data.sum(axis=1, numeric_only=True)
    
    ##--Add PCASP total counts to the ambient df--##
    FIREACE_particle_df_ambient['PCTcon'] = FIREACE_PCASP_sum_ambient.values
    
    ######################
    ##--Calc N(10-130)--##
    ######################
    
    ##--Calculate particles below UHSAS lower cutoff--##
    FIREACE_n_10_130 = (FIREACE_particle_df['CPC10_conc'] - 
                       FIREACE_particle_df['PCTcon'])
    
    ##--Change calculated particle counts less than zero to NaN--##
    FIREACE_n_10_130 = np.where(FIREACE_n_10_130 >= 0, FIREACE_n_10_130, np.nan)
    
    FIREACE_particle_df['n_10_130'] = FIREACE_n_10_130
    
    ##--Compute TOTAL counts from all size bins combined--##
    FIREACE_particle_df['Total_particles_STP'] = (FIREACE_particle_df['n_3_10'].fillna(0) + 
          FIREACE_particle_df['n_10_130'].fillna(0) + 
          FIREACE_particle_df['PCTcon'].fillna(0))
    
    ##--Calculate potential temperature--##
    FIREACE_ptemp = calc_ptemp(FIREACE_temperature, FIREACE_pressure)
    
    ##--Also calculate ptemp using nonaveraged data--##
    FIREACE_ptemp_nonav = calc_ptemp(FIREACE_temperature_nonav, FIREACE_pressure_nonav)
    
    ##--Repeat for ambient conditions--##
    FIREACE_n_10_130_ambient = (FIREACE_particle_df_ambient['CPC10_conc'] -
                               FIREACE_particle_df_ambient['PCTcon'])
    
    ##--Change calculated counts less than zero to NaN--##
    FIREACE_n_10_130_ambient = np.where(FIREACE_n_10_130_ambient >= 0, 
                                        FIREACE_n_10_130_ambient, np.nan)
    
    ##--Add ambient N_10_130 to the df--##
    FIREACE_particle_df_ambient['n_10_130'] = FIREACE_n_10_130_ambient
    
    ##--Compute the TOTAL counts from all bins--##
    FIREACE_particle_df_ambient['Total_particles'] = (FIREACE_particle_df_ambient['n_3_10'].fillna(0) +
            FIREACE_particle_df_ambient['n_10_130'].fillna(0) +
            FIREACE_particle_df_ambient['PCTcon'].fillna(0))
    
    ###################
    ##--Uncertainty--##
    ###################
    
    ##--For the nucleation mode--##
    
    ##--Pull R1 CPC data with zero periods still included from NETCARE--##
    ##--Identical set of instruments deployed for FIRE-ACE, but no zeros--##
    ##--First propagate through the STP conversion--##
    '''
    T_error = 0.3 # K, constant
    P_error = 100 + 0.0005*(FIREACE_pressure_nonav)
    
    
    ##--Use formula for mult/div to compute error after converting to STP--##
    CPC3_sigma_STP = (CPC3_sigma)((((P_error)/(FIREACE_pressure_nonav))**2 + 
                       ((T_error)/(FIREACE_temperature_nonav)**2))**(0.5))
    
    
    CPC10_sigma_STP = (CPC10_sigma)*((((P_error)/(FIREACE_pressure_nonav))**2 + 
                       ((T_error)/(FIREACE_temperature_nonav)**2))**(0.5))
    '''
    
    ##--Calculate the 3 sigma uncertainty for nucleating particles--##
    ##--Follow Williamson et al. Nature 2019 method--##
    ##--Uncertainty due to STP conversion is negligible, neglect--##
    nuc_error_3sigma_nonav = ((((CPC3_sigma)**2)*
                       (FIREACE_particle_df_nonav['CPC3_conc']) + 
                       ((CPC10_sigma)**2)*
                       (FIREACE_particle_df_nonav['CPC10_conc']))**(0.5))
    
    ##--Repeat for the 2 min averaged data--##
    nuc_error_3sigma = ((((CPC3_sigma)**2)*
                       (FIREACE_particle_df['CPC3_conc']) + 
                       ((CPC10_sigma)**2)*
                       (FIREACE_particle_df['CPC10_conc']))**(0.5))
    
    ###########################
    ##--Wrangle binned data--##
    ###########################
    
    ##--Append Nucleating and Aitken lists with dataframes--##
    ##--Use the NON-AVERAGED data for nulceation mode--##
    FIREACE_nucleating_df_nonav = pd.DataFrame({'nucleating': FIREACE_nuc_particles_nonav,
                              'nuc_error': nuc_error_3sigma_nonav,
                              'latitude': FIREACE_latitude_nonav, 
                              'PTemp': FIREACE_ptemp_nonav, 
                              'time': FIREACE_time_nonav}).set_index(FIREACE_time_nonav)
    
    ##--Build a nucleating df from the AVERAGED data, too--##
    FIREACE_nucleating_df = pd.DataFrame({'nucleating': FIREACE_nuc_particles, 
                                          'nuc_error': nuc_error_3sigma})
    
    ##--Add uncertainty to the existing averaged nucleation df--##
    FIREACE_particle_df['nuc_error'] = nuc_error_3sigma
    
    ##--Convert aitken mode data to dataframe--##
    FIREACE_aitken_df = pd.DataFrame({'aitken': FIREACE_n_10_130,
                              'nuc_error': nuc_error_3sigma,
                              'latitude': FIREACE_latitude, 
                              'PTemp': FIREACE_ptemp, 
                              'time': FIREACE_time}).set_index(FIREACE_time)
    
    
    ##--Also calculate the aitken mode at ambient conditions--##
    FIREACE_aitken_df_ambient = pd.DataFrame({'aitken': FIREACE_n_10_130_ambient,
                                              'latitude': FIREACE_latitude,
                                              'PTemp': FIREACE_ptemp,
                                              'time': FIREACE_time}).set_index(FIREACE_time)
    
    ##--Place all binned data in a single df--##
    FIREACE_all_bins_aligned = FIREACE_PCASP_STP
    FIREACE_all_bins_aligned['70'] = FIREACE_particle_df['n_10_130']
    
    ##--Ensure particle bin dataframes are indexed to time_index and properly named--##
    FIREACE_diameter_df = {}
    
    for col in FIREACE_all_bins_aligned.columns:
        FIREACE_diameter_df[str(col)] = pd.DataFrame(
            FIREACE_all_bins_aligned[col].values,
            index=FIREACE_all_bins_aligned.index, columns=[str(col)])
        
    ##--Repeat for ambient conditions--##
    FIREACE_all_bins_aligned_ambient = FIREACE_PCASP
    FIREACE_all_bins_aligned_ambient['70'] = FIREACE_particle_df_ambient['n_10_130']
    
    ##--Ensure dfs are indexed to time--##
    FIREACE_diameter_df_ambient = {}
    
    for col in FIREACE_all_bins_aligned_ambient.columns:
        FIREACE_diameter_df_ambient[str(col)] = pd.DataFrame(
            FIREACE_all_bins_aligned_ambient[col].values,
            index=FIREACE_all_bins_aligned_ambient.index, columns=[str(col)])
    
    ##--Append conditions list--##
    FIREACE_conditions_df = pd.DataFrame({'temperature': 
        FIREACE_temperature_nonav, 'pressure': FIREACE_pressure_nonav, 
        'PTemp': FIREACE_ptemp_nonav, 'RH': FIREACE_RH_nonav,
        'rBC':np.full(len(FIREACE_time_nonav), np.nan), 
        'O3':np.full(len(FIREACE_time_nonav), np.nan), 
        'latitude': FIREACE_latitude_nonav,
        'longitude': FIREACE_longitude_nonav,
        'altitude': FIREACE_altitude_nonav})
        
    ##--Make sure to have separate averaged conditions df--##
    FIREACE_avg_conditions_df = pd.DataFrame({'temperature':
         FIREACE_temperature, 'pressure': FIREACE_pressure, 
         'PTemp': FIREACE_ptemp, 'RH': FIREACE_RH, 
         'rBC': np.full(len(FIREACE_time), np.nan),
         'O3': np.full(len(FIREACE_time), np.nan),
         'latitude': FIREACE_latitude,
         'altitude': FIREACE_altitude})

    return {"FIREACE_nucleating_nonav": FIREACE_nucleating_df_nonav, 
            "FIREACE_nucleating" : FIREACE_nucleating_df,
            "FIREACE_aitken": FIREACE_aitken_df, 
            "FIREACE_aitken_ambient": FIREACE_aitken_df_ambient,
            "FIREACE_diameters": FIREACE_diameter_df,
            "FIREACE_diameters_ambient": FIREACE_diameter_df_ambient,
            "FIREACE_conditions": FIREACE_conditions_df, 
            "FIREACE_avg_conditions": FIREACE_avg_conditions_df,
            "FIREACE_PCASP_error": FIREACE_PCASP_error}

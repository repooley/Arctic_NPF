# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 16:25:28 2026

@author: repooley
"""

import pandas as pd
import icartt
import numpy as np
from pathlib import Path

'''
This script calculates the particle distribution for NETCARE
'''

###########################
##--Establish directory--##
###########################

##--Path to this script--##
script_path = Path(__file__).resolve()

##--Path to the root which is 3 levels up in the directory--##
root = script_path.parents[3]

##--Path to raw NETCARE data--##
directory = root / "NETCARE2015" / "data" / "raw"

####################################
##--Pull particle bin info files--##
####################################

##--Pull R1 data for the two CPC instruments - zeroes not yet filtered out--##
CPC3_R1 = icartt.Dataset(directory / "CPC_R1" / "CPC3776_Polar6_20150408_R1_L2.ict")
CPC10_R1 = icartt.Dataset(directory / "CPC_R1" / "CPC3772_Polar6_20150408_R1_L2.ict")

##--UHSAS bin data are in a CSV file--##
UHSAS_bins = pd.read_csv(directory / "NETCARE2015_UHSAS_bins.csv")

##--Make list of columns to pull, each named bin_x--##
##--Bins 1-13 not trustworthy. Bins 76-99 overlap with OPC, discard--##
##--Trim to use bins 14-76 (500>85 nm)--##
UHSAS_bin_num = [f'bin_{i}' for i in range(14, 75)]

##--Information for bins 14 thru 99--##
UHSAS_bin_center = UHSAS_bins['bin_avg'].iloc[14:75]
UHSAS_lower_bound = UHSAS_bins['lower_bound'].iloc[14:75]
UHSAS_upper_bound = UHSAS_bins['upper_bound'].iloc[14:75]

##--OPC bin data are in a CSV file--##
OPC_bin_info = pd.read_csv(directory / "NETCARE2015_OPC_bins.csv")

def calc_particle_bins(data):

    ##--Particle--##
    rBC = data["rBC"]
    CPC3 = data["CPC3"]
    CPC10 = data["CPC10"]
    UHSAS = data["UHSAS"]
    OPC = data["OPC"]
    
    ##--AIMMS data--##
    AIMMS = data["AIMMS"]
    
    time = AIMMS.data["TimeWave"]
    altitude = AIMMS.data["Alt"] # m
    temperature = AIMMS.data["Temp"] + 273.15 # K
    pressure = AIMMS.data["BP"] # pa
   
    ###############################
    ##--De-Normalize UHSAS Data--##
    ###############################
    
    ##--Pull UHSAS bin info--##
    UHSAS_cols = data["UHSAS_new_col_names"]
    UHSAS_upper_bounds = data["UHSAS_upper_bounds"]
    UHSAS_lower_bounds = data["UHSAS_lower_bounds"]
    UHSAS_bin_center = data["UHSAS_bin_center"]
    
    ##--Calculate dlogDp for UHSAS bins--##
    UHSAS_dlogDp = np.log(UHSAS_upper_bounds.values) - np.log(UHSAS_lower_bounds.values)
    
    ##--Get only particle count data (excluding 'Time')--##
    UHSAS_particle_counts = UHSAS.loc[:, UHSAS_cols] 
    
    ##--De-Normalize counts by multiplying by dlogDp across all rows--##
    UHSAS_denorm_counts = UHSAS_particle_counts.multiply(UHSAS_dlogDp, axis=1)
    
    ############################
    ##--Standardize OPC data--##
    ############################
    
    ##--Pull OPC bin info--##
    OPC_cols = data["OPC_new_col_names"]
    OPC_upper_bounds = data["OPC_upper_bounds"]
    OPC_bin_center = data["OPC_bin_center"]
    
    ##--Use the de-normalized values for calculating NPF--##
    
    ##--OPC samples every six seconds. Most rows are NaN--##
    ##--Forward-fill NaN values to propagate last valid reading--##
    ##--Limit forward filling to 5 NaN rows--##
    OPC_bins_filled = OPC.ffill(limit=5)
      
    ##--Get only particle count data (excluding 'Time')--##
    OPC_particle_counts = OPC_bins_filled.loc[:, OPC_cols]
    
    ##--Convert to STP!--##
    P_STP = 101325  # Pa
    T_STP = 273.15  # K
    
    ##--DENORM OPC--##
    OPC_conc_STP = []
    
    for OPC, T, P in zip(OPC_particle_counts.values, temperature, pressure):
        if np.isnan(T) or np.isnan(P):
            ##--Append with NaN if any input is NaN--##
            OPC_conc_STP.append([np.nan]*len(OPC))
        else:
            ##--Perform conversion if all inputs are valid--##
            corrected_OPC = OPC * (P_STP / P) * (T / T_STP)
            OPC_conc_STP.append(corrected_OPC)
            
    ##--Convert back to DataFrame with same columns and index--##
    OPC_conc_STP = pd.DataFrame(OPC_conc_STP, columns=OPC_particle_counts.columns, index=OPC_particle_counts.index)
    
    ######################
    ##--Calc N(2.5-10)--##
    ######################
    
    ##--Convert to STP!--##
    P_STP = 101325  # Pa
    T_STP = 273.15  # K
    
    ##--Create empty list for CPC3 particles--##
    CPC3_conc_STP = []
    
    for CPC3, T, P in zip(CPC3, temperature, pressure):
        if np.isnan(CPC3) or np.isnan(T) or np.isnan(P):
            ##--Append with NaN if any input is NaN--##
            CPC3_conc_STP.append(np.nan)
        else:
            ##--Perform conversion if all inputs are valid--##
            CPC3_conversion = CPC3 * (P_STP / P) * (T / T_STP)
            CPC3_conc_STP.append(CPC3_conversion)
        
    ##--Create empty list for CPC10 particles--##
    CPC10_conc_STP = []
    
    for CPC10, T, P in zip(CPC10, temperature, pressure):
        if np.isnan(CPC10) or np.isnan(T) or np.isnan(P):
            ##--Append with NaN if any input is NaN--##
            CPC10_conc_STP.append(np.nan)
        else:
            ##--Perform conversion if all inputs are valid--##
            CPC10_conversion = CPC10 * (P_STP / P) * (T / T_STP)
            CPC10_conc_STP.append(CPC10_conversion)
    
    ##--Creates a Pandas dataframe for particle data--##
    df = pd.DataFrame({'Altitude': altitude, 'CPC3_conc':CPC3_conc_STP, 'BC_mass': rBC,
                       'CPC10_conc': CPC10_conc_STP})
    
    ##--Calculate N3-10 particles--##
    nuc_particles = (df['CPC3_conc'] - df['CPC10_conc'])
    
    ##--Change calculated particle counts less than zero to NaN--##
    nuc_particles = np.where(nuc_particles >= 0, nuc_particles, np.nan)
    
    ##--Add nucleating particles to df--##
    df['nuc_particles'] = nuc_particles
    
    #####################
    ##--Calc N(10-89)--##
    #####################
    
    ##--Re-compute UHSAS total count using denormalized data--##
    UHSAS_total = UHSAS_denorm_counts.sum(axis=1)
    
    ##--Create df with UHSAS total counts and index to AIMMS time--##
    UHSAS_total_aligned = pd.DataFrame({'Time': time, 'Total_count': UHSAS_total}).set_index('Time')
    
    ##--Same for OPC--##
    OPC_total = OPC_conc_STP.sum(axis=1)
    
    OPC_total_aligned = pd.DataFrame({'Time': time, 'Total_count': OPC_total}).set_index('Time')
    
    ##--Create df with CPC10 counts and set index to time--##
    CPC10_counts = pd.DataFrame({'Time':time, 'Counts':CPC10_conc_STP}).set_index('Time')
    
    ##--Calculate particles below UHSAS lower cutoff--##
    n_10_89 = (CPC10_counts['Counts'] - (UHSAS_total_aligned['Total_count'] + OPC_total_aligned['Total_count']))
    
    ##--Change calculated particle counts less than zero to NaN--##
    n_10_89 = np.where(n_10_89 >= 0, n_10_89, np.nan)
    
    ##--Add 10-89 nm particles to the dataframe--##
    df['n_10_89'] = n_10_89
    
    ##--Specify n_10_89 bin center as variable--##
    n_10_89_center = 49.5
    
    ##--Convert the bin center to a series--##
    n_10_89_center = pd.Series(n_10_89_center)
    
    ###########################
    ##--Wrangle binned data--##
    ###########################
    
    ##--Concatenate bin edges--##
    combined_bin_edges = np.concatenate([
        [2.5],      # start of first bin
        [10],       # upper edge of N(2.5-10), also lower of next
        [89.32],       # upper edge of N(10-89), also lower of next
        UHSAS_upper_bound.values,  # UHSAS bins continue from 85
        OPC_upper_bounds.values     # OPC bins continue from last UHSAS
    ])
    
    ##--Concatenate bin centers and reindex--##
    bin_centers = pd.concat([n_10_89_center, UHSAS_bin_center, OPC_bin_center], axis=0).reset_index(drop=True)
    
    ##--Pull aligned UHSAS bin data--##
    UHSAS_aligned = data["UHSAS"]
    
    ##--Convert to a df--##
    UHSAS_aligned = pd.DataFrame(UHSAS_aligned)
    
    ##--Convert n_10_89 to a df--##
    n_10_89_df = pd.DataFrame(n_10_89)
    
    ##--Place all binned data in a single df--##
    all_bins_aligned = pd.concat([n_10_89_df, UHSAS_aligned, OPC_conc_STP], axis=1)
    total_particle_count = all_bins_aligned.sum(axis=1, numeric_only=True) 
    
    ##--Add total count to df--##
    df["total_count"] = total_particle_count
    
    ##--Create a dictionary to store each column as a separate dataframe, col names are keys--##
    diameter_dfs = {col: pd.DataFrame({col: all_bins_aligned[col]}) for col in all_bins_aligned.columns}
    
    #############################
    ##--Calculate Uncertainty--##        
    #############################
    
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
    
    ##--UHSAS doesn't have zero periods, using Poisson counting uncertainty--##
    UHSAS_total_sqrt = np.sqrt(UHSAS_denorm_counts)
    
    ##--Use simple sum of UHSAS uncertainties per bin for conservative estimate--##
    ##--Similar result as using sqrt of squares but erring on side of caution--##
    UHSAS_total_error = UHSAS_total_sqrt.sum(axis=1)
    
    ##--Repeat for OPC data--##
    OPC_total_sqrt = np.sqrt(OPC_conc_STP)
    
    OPC_total_error = OPC_total_sqrt.sum(axis=1)
    
    #############################
    ##--Propagate uncertainty--##
    #############################
    
    ##--The ICARTT files for CPC instruments say 10% uncertainty of meas value - feels conservative for large counts!--##
    ##--Calculate the 3 sigma uncertainty for nucleating particles--##
    
    T_error = 0.3 # K, constant
    P_error = 100 + 0.0005*(pressure)
    
    ##--Use formula for mult/div to compute error after converting to STP--##
    greater3nm_error = ((CPC3_conc_STP)*(((P_error)/(pressure))**2 + 
                       ((T_error)/(temperature))**2 + 
                       ((CPC3_sigma)/(CPC3_conc_STP)))**(0.5))
    
    greater10nm_error = ((CPC10_conc_STP)*(((P_error)/(pressure))**2 + 
                        ((T_error)/(temperature))**2 + 
                        ((CPC10_sigma)/(CPC10_conc_STP)))**(0.5))
    
    ##--Use add/subtract forumula to compute 3sigma error--##
    nuc_error_3sigma = (((greater3nm_error)**2 + (greater10nm_error)**2)**(0.5))*3
    
    ##--nuc_error_3sigma still has a time index, reset to integer to align--##
    df['nuc_error_3sigma'] = nuc_error_3sigma
    
    ##--Calculate error in difference between CPC10 and UHSAS + OPC--##
    aitken_error_3sigma = (((greater10nm_error)**2 + (UHSAS_total_error)**2 + (OPC_total_error)**2)**(0.5))*3
    
    ##--Add uncertainty for 10-85 nm bin to big df--##
    df['aitken_error_3sigma'] = aitken_error_3sigma
    
    return {'df': df, 'diameter_dfs': diameter_dfs}

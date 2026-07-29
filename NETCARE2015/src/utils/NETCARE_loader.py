# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 13:38:14 2026

@author: repooley
"""

import os
import glob
import pandas as pd
import icartt
from datetime import date
import numpy as np
from pathlib import Path

'''
This script reads in all data for the NETCARE campaign and aligns it to the
AIMMS instrument, which is the timekeeper.
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


##############################
##--Pull particle bin info--##
##############################

##--UHSAS--##

##--Read in CSV file with bin boundaries for the UHSAS--##
UHSAS_bins = pd.read_csv(directory / "NETCARE2015_UHSAS_bins.csv")

##--Make list of columns to pull, each named bin_x--##
##--Bins 1-13 not trustworthy. Bins 76-99 overlap with OPC, discard--##
##--Trim to use bins 14-76 (500>85 nm)--##
UHSAS_bin_num = [f'bin_{i}' for i in range(14, 75)]

##--Information for bins 14 thru 99--##
UHSAS_bin_center = UHSAS_bins['bin_avg'].iloc[14:75]
UHSAS_lower_bounds = UHSAS_bins['lower_bound'].iloc[14:75]
UHSAS_upper_bounds = UHSAS_bins['upper_bound'].iloc[14:75]

##--OPC--##

##--Read in CSV file with bin boundaries for the OPC--##
OPC_bins = pd.read_csv(directory / "NETCARE2015_OPC_bins.csv")

##--Select bins greater than 500 nm (Channel 7 and greater)--##
OPC_bin_center = OPC_bins['bin_avg'].iloc[6:31]
OPC_lower_bounds = OPC_bins['lower_bound'].iloc[6:31]
OPC_upper_bounds = OPC_bins['upper_bound'].iloc[6:31]

##--Make list of columns to pull, each named Channel_x--##
OPC_bin_num = [f'Channel_{i}' for i in range(7, 32)]

########################################
##--Tie flight number to flight date--##
########################################

flight_dates = {"Flight1": date(2015, 4, 5),
    "Flight2": date(2015, 4, 7),
    "Flight3": date(2015, 4, 8),
    "Flight4": date(2015, 4, 8),
    "Flight5": date(2015, 4, 9),
    "Flight6": date(2015, 4, 11),
    "Flight7": date(2015, 4, 13),
    "Flight8": date(2015, 4, 20),
    "Flight9": date(2015, 4, 20),
    "Flight10": date(2015, 4, 21)}

##--Find the files for every flight called--##
def find_files(directory, flight, partial_name):
    search_pattern = os.path.join(directory, flight, f"*{partial_name}*")
    return sorted(glob.glob(search_pattern))

##--Align each file to AIMMS time--##
def align_to_aimms(reference_time, time_array, data_array):
    df = pd.DataFrame({"time": time_array, "value": data_array})
    aligned = df.set_index("time").reindex(reference_time)
    return aligned["value"]

##--Load in every dataset and supplemental data--##
def load_flight(directory, flight):
    
    #####################
    ##--Open datasets--##
    #####################
    
    ##--Ensure only one file is pulled for each, except for O3--##
    
    ##--AIMMS: time, aircraft location, temp, RH--##
    aimms = icartt.Dataset(find_files(directory, flight, "AIMMS_POLAR6")[0])
    
    ##--Pull 
    
    ##--Particle sizing datasets--##
    CPC3 = icartt.Dataset(find_files(directory, flight, "CPC3776")[0])
    CPC10 = icartt.Dataset(find_files(directory, flight, "CPC3772")[0])
    UHSAS = icartt.Dataset(find_files(directory, flight, "UHSAS")[0])
    OPC = icartt.Dataset(find_files(directory, flight, "OPC")[0])

    ##--Black carbon observations--##
    SP2 = icartt.Dataset(find_files(directory, flight, "SP2_Polar6")[0])
    
    ##--Trace gas observations--##
    CO = icartt.Dataset(find_files(directory, flight, "CO_POLAR6")[0])
    CO2 = icartt.Dataset(find_files(directory, flight, "CO2_POLAR6")[0])
    H2O = icartt.Dataset(find_files(directory, flight, "H2O_POLAR6")[0])
    O3_files = find_files(directory, flight, "O3_POLAR6")
    
    ##--Set AIMMS time as the reference--##
    aimms_time = aimms.data["TimeWave"]
    
    ##--Pull the date for the requested flight--##
    flight_date = flight_dates[flight]

    #########################
    ##--Align instruments--##
    #########################
    
    ##--Particle sizing--##
    
    ##--CPC3--##
    CPC3_time = CPC3.data["time"]
    CPC3_conc = CPC3.data["conc"]
    CPC3_aligned = align_to_aimms(aimms_time, CPC3_time, CPC3_conc)

    ##--CPC10--##
    CPC10_time = CPC10.data["time"]
    CPC10_conc = CPC10.data["conc"]
    CPC10_aligned = align_to_aimms(aimms_time, CPC10_time, CPC10_conc)

    ##--UHSAS--##
    UHSAS_time = UHSAS.data["time"]
    ##--Put column names and content in a dictionary and then convert to a Pandas df--##
    UHSAS_bins = pd.DataFrame({col: UHSAS.data[col] for col in UHSAS_bin_num})
    ##--Create new column names by rounding the bin center values to the nearest integer--##
    UHSAS_new_col_names = UHSAS_bin_center.round().astype(int).tolist()
    ##--Rename the UHSAS_bins df columns to bin average values--##
    UHSAS_bins.columns = UHSAS_new_col_names
    ##--Add time back in as a column--##
    UHSAS_bins.insert(0, "Time", UHSAS_time)
    ##--Reindex--##
    UHSAS_bins_aligned = UHSAS_bins.set_index("Time").reindex(aimms_time)
    
    ##--OPC--##
    OPC_time = OPC.data["Time_UTC"]
    ##--Put column names and content in a dictionary and then convert to a Pandas df--##
    OPC_bins = pd.DataFrame({col: OPC.data[col] for col in OPC_bin_num})
    ##--Create new column names by rounding the bin center values to the nearest integer--##
    OPC_new_col_names = OPC_bin_center.round().astype(int).tolist()
    ##--Rename the OPC_bins df columns to bin average values--##
    OPC_bins.columns = OPC_new_col_names
    ##--Add time back in as a column--##
    OPC_bins.insert(0, "Time", OPC_time)
    ##--Reindex--##
    OPC_bins_aligned = OPC_bins.set_index("Time").reindex(aimms_time)

    
    ##--SP2--##
    BC_time = SP2.data["Time_UTC"]
    BC_mass = SP2.data["BC_mass_concSTP"] * 1000
    BC_mass_aligned = align_to_aimms(aimms_time, BC_time, BC_mass)
    
    ##--Trace gases--##
    
    ##--CO--##
    CO_time = CO.data["Time_UTC"]
    CO_conc = CO.data["CO_ppbv"]
    CO_aligned = align_to_aimms(aimms_time, CO_time, CO_conc)
    
    ##--CO2--##
    CO2_time = CO2.data["Time_UTC"]
    CO2_conc = CO2.data["CO2_ppmv"]
    CO2_aligned = align_to_aimms(aimms_time, CO2_time, CO2_conc)
    
    ##--H2O--##
    H2O_time = H2O.data["Time_UTC"]
    H2O_conc = H2O.data["H2O_ppmv"]
    H2O_aligned = align_to_aimms(aimms_time, H2O_time, H2O_conc)
    
    #####################
    ##--Calc humidity--##
    #####################

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
    for T, P, H2O_ppmv in zip(aimms.data['Temp'], aimms.data['BP'], H2O_aligned):
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

    ##--Place in dataframe for use--##
    RH = pd.DataFrame({'RH': relative_humidity_w})

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
    for T, P, H2O_ppmv in zip(aimms.data['Temp'], aimms.data['BP'], H2O_aligned):
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
    
    ##--O3--##
    ##--Flight 2 requires concatenation for O3 files--##
    
    ##--Store concat data here--##
    all_times = []
    all_conc = []
    
    ##--loop through O3 files and concat--##
    for file in O3_files:
        O3 = icartt.Dataset(file)
        all_times.extend(O3.data['Start_UTC'])
        all_conc.extend(O3.data['O3'])
    
    ##--Convert O3 concat files to np arrays--##
    O3_starttime = np.array(all_times)
    O3_conc = np.array(all_conc)
    
    ##--Pull time as seconds since midngight--##
    O3_seconds_since_midnight = O3_starttime.astype(int)

    ##--Put in dataframe so duplicates can be removed--##
    O3_df = pd.DataFrame({'Time_UTC': O3_seconds_since_midnight,
        'O3': O3_conc})
    
    ##--Remove any overlapping timestamps--##
    O3_df = O3_df.drop_duplicates(subset='Time_UTC')
    
    ##--Align O3 (lower-res) to AIMMS time, will fill gaps with NaNs--##
    O3_aligned = O3_df.set_index('Time_UTC').reindex(aimms_time)
    
    ##--Function outputs to be used in analyses--##
    return {"flight_date": flight_date,
        "AIMMS": aimms,
        "CPC3": CPC3_aligned,
        "CPC10": CPC10_aligned,
        "UHSAS": UHSAS_bins_aligned,
        "UHSAS_bin_center": UHSAS_bin_center,
        "UHSAS_upper_bounds": UHSAS_upper_bounds,
        "UHSAS_lower_bounds": UHSAS_lower_bounds,
        "UHSAS_new_col_names": UHSAS_new_col_names,
        "OPC": OPC_bins_aligned,
        "OPC_bin_center": OPC_bin_center,
        "OPC_upper_bounds": OPC_upper_bounds,
        "OPC_lower_bounds": OPC_lower_bounds,
        "OPC_new_col_names": OPC_new_col_names,
        "rBC": BC_mass_aligned,
        "CO": CO_aligned,
        "CO2": CO2_aligned, 
        "H2O": H2O_aligned, 
        "RH": RH,
        "O3": O3_aligned}
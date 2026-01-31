# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 14:31:27 2026

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
from datetime import date

###################
##--User inputs--##
###################

##--Set the base directory to project folder--##
directory = r"C:\Users\repooley\REP_PhD\Arctic_NPF\PAMARCMiP2012\data"

##--Select flight (Flight1 thru Flight9)--##
flights_to_analyze = ["Flight1", "Flight2", "Flight3", "Flight6", 
                      "Flight7", "Flight8", "Flight9"]

##--Bin data are in a CSV file--##
##--SAME bins for NETCARE and PAMARCMiP--##
UHSAS_bins = pd.read_csv(r"C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data\raw\NETCARE2015_UHSAS_bins.csv")

#########################
##--Open ICARTT Files--##
#########################

##--Define function that creates datasets from filenames--##
def find_files(directory, flight, partial_name):
    ##--flight data are stored in a folder called "raw"--##
    flight_dir = os.path.join(directory, "raw", flight)
    search_pattern = os.path.join(flight_dir, f"*{partial_name}*")
    return sorted(glob.glob(search_pattern))

for flight in flights_to_analyze:

    ##--Pull file--##
    data = pd.read_csv(find_files(directory, flight, ".csv")[0])
    
    #################
    ##--Pull data--##
    #################
    
    ##--Data--##
    altitude = data['Altitude'] # in m
    latitude = data['Latitude'] # in degrees
    temperature = data['Temp'] + 273.15 # in K
    pressure = data['Pressure'] # in pa
    time = data['Time'] # seconds since midnight
    
    ##--The first datapoint in 'latitude' column is erraneous (47.12 N)--##
    latitude = latitude.where(latitude >= 50, np.nan)
    
    ##--USHAS Data--##
    UHSAS_total_num = data['UH-TotConc'] # particles/cm^3
    
    ##--Bin data are in a CSV file--##
    ##--SAME bins for NETCARE and PAMARCMiP--##
    UHSAS_bins = pd.read_csv(r"C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data\raw\NETCARE2015_UHSAS_bins.csv")
    
    ##--Make list of columns to pull, each named bin_x--##
    UHSAS_bin_num = [f'UH_bin_{i}' for i in range(1, 100)]
    
    ##--Information for bins 1 thru 99--##
    UHSAS_bin_center = UHSAS_bins['bin_avg'].iloc[0:100]
    UHSAS_lower_bound = UHSAS_bins['lower_bound'].iloc[0:100]
    UHSAS_upper_bound = UHSAS_bins['upper_bound'].iloc[0:100]
    
    ##--Put column names and content in a dictionary and then convert to a Pandas df--##
    UHSAS_bins = pd.DataFrame({col: data[col] for col in UHSAS_bin_num})
    
    ##--Create new column names by rounding the bin center values to the nearest integer--##
    UHSAS_new_col_names = UHSAS_bin_center.round().astype(int).tolist()
    
    ##--Rename the UHSAS_bins df columns to bin average values--##
    UHSAS_bins.columns = UHSAS_new_col_names
    
    ##--10 nm CPC data--##
    CPC10_conc = data['CPC10'] # count/cm^3
    
    ##--REMOVE OUTLIERS above 99th percentile--##
    p = 0.99
    
    ##--Compute threshold for each UHSAS column--##
    uhsas_thresh = UHSAS_bins.quantile(p)
    
    ##--keep only rows where each bin is below its threshold--##
    UHSAS_bins_filtered = UHSAS_bins[UHSAS_bins.le(uhsas_thresh, axis=1)]
    
    cpc10_thresh = CPC10_conc.quantile(p)
    CPC10_filtered = CPC10_conc[CPC10_conc <= cpc10_thresh]
    
    ####################################
    ##--Assign date to flight number--##
    ####################################
    
    if flight=="Flight1":
        flight_date = date(2012, 3, 29)
    elif flight=="Flight2":
        flight_date = date(2012, 3, 30)
    elif flight=="Flight3":
        flight_date = date(2012, 4, 2)
    elif flight=="Flight4" or flight=="Flight5":
        flight_date = date(2012, 4, 3)
    elif flight=="Flight6":
        flight_date = date(2012, 4, 4)
    elif flight=="Flight7": 
        flight_date = date(2012, 4, 5)
    elif flight=="Flight8": 
        flight_date = date(2012, 4, 6)
    elif flight=="Flight9": 
        flight_date = date(2012, 4, 7)
    
    ###############################
    ##--De-Normalize UHSAS Data--##
    ###############################
    
    ##--For total count calculation--##
    
    ##--Calculate dlogDp for UHSAS bins--##
    UHSAS_dlogDp = np.log(UHSAS_upper_bound.values) - np.log(UHSAS_lower_bound.values)
    
    ##--Get only particle count data (excluding 'Time')--##
    UHSAS_particle_counts = UHSAS_bins.loc[:, UHSAS_new_col_names]  # Adjust column names as needed
    
    ##--De-Normalize counts by multiplying by dlogDp across all rows--##
    UHSAS_denorm_counts = UHSAS_particle_counts.multiply(UHSAS_dlogDp, axis=1)
    
    ##################
    ##--UHSAS bins--##
    ##################
 
    ##--Bin data are in a CSV file--##
    ##--SAME bins for NETCARE and PAMARCMiP--##
    UHSAS_bins = pd.read_csv(r"C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data\raw\NETCARE2015_UHSAS_bins.csv")
    
    ##--Make list of columns to pull, each named bin_x--##
    UHSAS_bin_num = [f'UH_bin_{i}' for i in range(1, 100)]
    
    ##--Information for bins 1 thru 99--##
    UHSAS_bin_center = UHSAS_bins['bin_avg'].iloc[0:100]
    UHSAS_lower_bound = UHSAS_bins['lower_bound'].iloc[0:100]
    UHSAS_upper_bound = UHSAS_bins['upper_bound'].iloc[0:100]
    
    ##--Put column names and content in a dictionary and then convert to a Pandas df--##
    UHSAS_bins = pd.DataFrame({col: data[col] for col in UHSAS_bin_num})
    
    ##--Create new column names by rounding the bin center values to the nearest integer--##
    UHSAS_new_col_names = UHSAS_bin_center.round().astype(int).tolist()
    
    ##--Rename the UHSAS_bins df columns to bin average values--##
    UHSAS_bins.columns = UHSAS_new_col_names
    
    ##--REMOVE OUTLIERS above 99th percentile--##
    p = 0.99
    
    ##--Compute threshold for each UHSAS column--##
    uhsas_thresh = UHSAS_bins.quantile(p)
    
    ##--keep only rows where each bin is below its threshold--##
    UHSAS_bins_filtered = UHSAS_bins[UHSAS_bins.le(uhsas_thresh, axis=1)]
 
    #####################
    ##--Calc N(10-60)--##
    #####################
    
    ##--Create df with UHSAS total counts--##
    UHSAS_total = pd.DataFrame({'Time': time, 'Total_count': UHSAS_total_num})
    
    ##--Create df with CPC10 counts and set index to time--##
    CPC10_counts = pd.DataFrame({'Time':time, 'Counts':CPC10_filtered})
    
    ##--Calculate particles below UHSAS lower cutoff--##
    n_10_60 = (CPC10_counts['Counts'] - UHSAS_total['Total_count'])
    
    ##--Change calculated particle counts less than zero to NaN--##
    n_10_60 = np.where(n_10_60 >= 0, n_10_60, np.nan)
    
    ##--Put N(10-60) bin center in a df--##
    n_10_60_center = pd.DataFrame([35])
    
    ##--Flatten--##
    n_10_60_center = pd.Series(n_10_60_center.values.flatten())
    
    ##--Convert n_10_60 to a df--##
    n_10_60_df = pd.DataFrame({'35': n_10_60, 'time':time}).set_index('time')
    
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
    
    ###########################
    ##--Wrangle binned data--##
    ###########################
    
    ##--Concatenate bin edges--##
    combined_bin_edges = np.concatenate([
        [10],       # start of first bin
        [60],       # upper edge of N(10-60), also lower of next
        UHSAS_upper_bound.values,  # UHSAS bins continue from 60
    ])
    
    time_averaged = data['Time']
    
    ##--Calculate time edges for each bin--##
    time_step = time_averaged.iloc[1] - time_averaged.iloc[0]  
    time_edges = np.append(time_averaged, time_averaged.iloc[-1] + time_step)  # length N + 1
    
    ##--Concatenate bin centers and reindex--##
    bin_centers = pd.concat([n_10_60_center, UHSAS_bin_center], axis=0).reset_index(drop=True)
    
    ##--Place all binned data in a single df--##
    all_bins_aligned = UHSAS_bins_filtered
    all_bins_aligned['35'] = n_10_60
    
    time_index = data['Time']  # use the same index as coagulation_sink
    
    ##--Ensure particle bin dataframes are indexed to time_index and properly named--##
    diameter_dfs = {}
    for col in all_bins_aligned.columns:
        vals = all_bins_aligned[col].to_numpy()
        # explicitly name the column as the bin diameter
        diameter_dfs[col] = pd.DataFrame({str(col): vals}, index=time_index[-len(vals):])
       
    ######################################
    ##--Condensation sink calculations--##
    ######################################
    
    ##--USING ABSOLUTE COUNT DATA--##
    
    ##--Constants--##
    
    R = 8.314 # Ideal gas constant (m^3*Pa*K^-1*mol^-1)
    ##--H2SO4 kinetic diam: lifted from Williamson et al for now (avg of their values)--##
    Ds = 5.49E-10 # in m
    ##--Kinetic diam of air calculated from mixing ratios and dataset on Wikepedia--##
    Dair = 3.61E-10 # in m
    avg_diam = (Ds + Dair)/2
    ##--Mass sulfuric acid--##
    Ms = 98.079 # g/mol
    ##--Mass air--##
    Mair = 28.96 # g/mol
    ##--Reduced mass--##
    Z = Ms/Mair 
    ##--Sticking coefficient - fair to assume unity for H2SO4--##
    alpha = 1
    ##--Boltzmann--##
    k = 1.38E-23 # J/K
    ##--Sutherland's law for dynamic viscosity--##
    C = 1.458E-6 # kg/ms*sqrt(K)
    S = 110.4 # K
    
    ##--Variables--##
    
    ##--Convert temperature and pressure from numpy array to dataframe to subvert errors--##
    ##--Convert temperature and pressure to series with aligned index--##
    temperature_series = pd.Series(temperature.values, index=time_index[-len(temperature):])
    pressure_series = pd.Series(pressure.values, index=time_index[-len(pressure):])
    
    Latitude_series = pd.Series(latitude, index=time_index)
    
    ##--Loop through dfs in diameter_dfs and calculate needed variables for each bin--##
    ##--Store in series initialized at zero--##
    condensation_sink = pd.Series(0, index=time_index)  
      
    
    for diameter, df in diameter_dfs.items():
        
        ##--Convert column diams from string to float--##
        mean_diameter = (float(diameter)) * 1E-9 # in m
        
        ##--Calculate mean free path of H2SO4 from molecular diameter--##
        df['mean_free_path'] = ((R * temperature_series.loc[df.index]) / ((2 ** (1/2)) 
                                * 3.14159 * (Ds ** 2) * 6.022E23 * pressure_series.loc[df.index])) # in m/molecule
        
        ##--Calculate the Knudsen number--##
        df['Knudsen_num'] = df['mean_free_path'] / (mean_diameter / 2) # unitless ratio
        
        ##--Calculate Fuch's correction--##
        df['Fuchs_correction'] = (1 + df['Knudsen_num']) / (1 + ((4/(3*alpha)) 
                                + 0.337) * df['Knudsen_num'] + (4/(3*alpha) * (df['Knudsen_num']) ** 2)) # unitless
        
        ##--Calculate slip correction for Dynamic Viscosity calculation--##
        df['Slip_correction'] = (1 + df['Knudsen_num'] * (2.514 + 0.800 * (np.exp(-0.550 / df['Knudsen_num'])))) # unitless
        
        ##--Calculate dynamic viscosity using Sutherland's law with constants for air--##
        df['Dynamic_viscosity'] = (C * (temperature_series.loc[df.index]) ** (3/2)) / (temperature_series.loc[df.index] + S)
        
        ##--Calculate the diffusion coefficient--##
        df['Diffusion_coefficient'] = ((k * temperature_series.loc[df.index] * df['Slip_correction']) / 
                                       (3 * 3.14159 * df['Dynamic_viscosity'] * Ds)) # m^2/s
        
        ##--Extract Particle Concentration (first column in diameter_dfs)--##
        df['Particle_concentration'] = df.iloc[:, 0] / 1E-6 # converted to #/m^3
        
        ##--Per-bin contribution to condensation sink (before final multiplication)--##
        df['CS_contribution'] = (df['Fuchs_correction'] * mean_diameter * df['Particle_concentration'])
        
        ##--Multiply each bin’s CS contribution by its diffusion coefficient--##
        ##--Fill NaN values in CS_contribution with zeros to prevent NaN result--##
        condensation_sink += (2 * np.pi * df['Diffusion_coefficient'] * df['CS_contribution']).fillna(0)
     
    ##--Populate series--##
    condensation_sink = pd.DataFrame({'Condensation_Sink': condensation_sink}) 
    
    #####################################
    ##--Calculate std dev from zeroes--##        
    #####################################
    
    ##--Pull datasets with zeros not filtered out--##
    CPC10_R1 = icartt.Dataset(r'C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data\raw\CPC_R1\CPC3772_Polar6_20150408_R1_L2.ict')
    
    ##--Pull CPC data from R1 data--##
    CPC10_R1_conc = CPC10_R1.data['conc']
    
    ##--Isolate zero periods, setting conservative upper limit of 50 counts--##
    ##--Numpy doesn't recognize -9999 as NaN, tell it to ignore these values--##
    CPC10_zeros_c = CPC10_R1_conc[(CPC10_R1_conc < 50) & (CPC10_R1_conc != -99999)]
    
    ##--Calculate standard deviation of zeros--##
    # Use ddof=1 for sample standard deviation
    CPC10_sigma = np.std(CPC10_zeros_c, ddof=1)
    
    greater10nm_error = 3*CPC10_sigma
    
    ##--UHSAS doesn't have zero periods, using Poisson counting uncertainty--##
    UHSAS_total_sqrt = np.sqrt(UHSAS_denorm_counts)
    
    ##--Use simple sum of UHSAS uncertainties per bin for conservative estimate--##
    ##--Similar result as using sqrt of squares but erring on side of caution--##
    UHSAS_total_error = UHSAS_total_sqrt.sum(axis=1)
    
    ##--Calculate error in difference between CPC10 and UHSAS + OPC--##
    aitken_error_3sigma = (((greater10nm_error)**2 + (UHSAS_total_error)**2)**(0.5))*3
    
    aitken_error_3sigma = pd.DataFrame({'aitken_error_3sigma': aitken_error_3sigma, 'time': time}).set_index('time')
    
    #######################################
    ##--Filter to NPF and non-NPF times--##
    #######################################
    
    condensation_n_10_60 = pd.DataFrame({'Condensation': condensation_sink['Condensation_Sink'], 'Aitken': n_10_60_df['35'],
                                     'LoD': aitken_error_3sigma['aitken_error_3sigma'], 'PTemp': potential_temp})
    
    condensation_aitken = condensation_n_10_60['Condensation'][condensation_n_10_60['Aitken'] > condensation_n_10_60['LoD']]
    condensation_noaitken = condensation_n_10_60['Condensation'][condensation_n_10_60['Aitken'] <= condensation_n_10_60['LoD']]
    conden_df = {'Aitken': condensation_aitken, 'No Aitken': condensation_noaitken}
    
    #############
    ##--Stats--##
    #############
    
    ##--Counts--##
    conden_aitken_count = len(condensation_aitken)
    conden_noaitken_count = len(condensation_noaitken)
    
    ##--Statistical signficance for unpaired non-parametric data: Mann-Whitney U test--##
    conden_npf_array = condensation_aitken.dropna().to_numpy() # data should be in a list or array
    conden_nonpf_array = condensation_noaitken.dropna().to_numpy()
    U_conden, p_conden = mannwhitneyu(conden_npf_array, conden_nonpf_array)
    
    ##--Calculate Z-score--##
    ##--Referenced https://datatab.net/tutorial/mann-whitney-u-test--##
    z_conden = (U_conden - conden_aitken_count*conden_noaitken_count/2)/((conden_aitken_count*
                conden_noaitken_count*(conden_aitken_count + conden_noaitken_count + 1)/12)**(1/2))
    
    ##--Take absolute value of Z scores--##
    z_conden = abs(z_conden)
    
    ##--Use Z-score to calculate rank biserial correlation, r--##
    r_conden = z_conden/((conden_aitken_count + conden_noaitken_count)**(1/2))
    
    ################
    ##--Plotting--##
    ################
    
    ##--Assign pallettes--##
    palette = {'Aitken':'#2f6794', 'No Aitken':'#1e537e'}

    fig, ax = plt.subplots(figsize = (4,6))
    ##--Cut=0 disallows interpolation beyond the data extremes--##
    condensation_plot = sns.violinplot(data=conden_df, palette=palette,
                                       inner_kws={'whis_width': 0, 'solid_capstyle':'butt'}, ax=ax, cut=0)
    ax.set(xlabel='')
    ax.set(ylabel='Condensation Sink (S-1)')
     
    ax.set(title=f"Condensation Sink - PAMARCMiP {flight.replace('Flight', 'Flight ')} ({flight_date})")
        
    ##--Add text labels with N--##
    plt.text(0.25, 0.12, "N={}".format(conden_aitken_count), transform=fig.transFigure, fontsize=10, color='dimgrey')
    plt.text(0.63, 0.12, "N={}".format(conden_noaitken_count), transform=fig.transFigure, fontsize=10, color='dimgrey')
    
    ##--Conditions for adding p values--##
    if p_conden >= 0.05:
        plt.text(0.33, 0.855, f"p={p_conden:.4f},", transform=fig.transFigure, fontsize=10, color='dimgrey')
    elif 0.05 > p_conden >= 0.0005:
        plt.text(0.33, 0.855, f"p={p_conden:.4f},", transform=fig.transFigure, fontsize=10, color='dimgrey')
    elif p_conden < 0.0005: 
        plt.text(0.33, 0.855, "p<0.0005,", transform=fig.transFigure, fontsize=10, color='dimgrey')
        
    ##--Add r value next to p-value--##
    plt.text(0.525, 0.855, f"r={r_conden:.3f}", transform=fig.transFigure, fontsize=10, color='dimgrey')
     
    #plt.savefig(f"{output_path}\\Sinks\condensation\conden_{flight}", dpi=600)
    
    plt.show()

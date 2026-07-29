# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 13:20:04 2026

@author: repooley
"""

import icartt
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from datetime import date
import seaborn as sns
from scipy.stats import mannwhitneyu

###################
##--User inputs--##
###################

##--Set the base directory to project folder--##
directory = r"C:\Users\repooley\REP_PhD\Arctic_NPF\ATom2018\data"

##--The flight10 file also includes flight11--##
flights_to_analyze = ["Flight2", "Flight10", "Flight11", "Flight12"]

##--Set binning for PTemp and Latitude--##
num_bins_lat = 10
num_bins_ptemp = 10

##--Define function that creates datasets from filenames--##
def find_files(directory, flight):
    ##--flight data are stored in a folder called "raw"--##
    flight_dir = os.path.join(directory, "raw", flight)
    search_pattern = os.path.join(flight_dir, "*.ict")
    return sorted(glob.glob(search_pattern))

nuc_dfs = []
aitken_dfs = []

##--Loop through each flight in the list--##
for flight in flights_to_analyze:
    
    #########################
    ##--Open ICARTT Files--##
    #########################

    dataset = icartt.Dataset(find_files(directory, flight)[0])

    ####################################
    ##--Assign date to flight number--##
    ####################################
    
    if flight=="Flight2":
        flight_date = date(2018, 4, 27)
    elif flight=="Flight10":
        flight_date = date(2018, 4, 17)
    elif flight=="Flight11": 
        flight_date = date(2018, 5, 18)
    elif flight=="Flight12":
        flight_date = date(2018, 5, 19)
    
    #################
    ##--Pull data--##
    #################
    
    ##--Ambient Data--##
    altitude = dataset.data['G_ALT'] # in m 
    latitude = dataset.data['G_LAT'] # in degrees
    temperature = dataset.data['T'] # in K
    pressure = dataset.data['P'] * 100 # in Pa
    
    ##--Hydrogen oxides from ATHOS--##
    OH = dataset.data['OH_ATHOS'] # pptv
    HO2 = dataset.data['HO2_ATHOS'] # pptv
    
    ##--CIT-CIMS (CF3O-) for peroxides and acids--##
    H2O2 = dataset.data['H2O2_CIT'] # pptv
    HNO3 = dataset.data['HNO3_CIT'] # pptv
    SO2 = dataset.data['SO2_CIT'] # pptv
    
    ##--NOAA Picarro--##
    CO = dataset.data['CO_NOAA'] # ppb
    CO2 = dataset.data['CO2_NOAA'] # ppm
    CH4 = dataset.data['CH4_NOAA'] # ppb
    
    CO_CO2 = CO / CO2 # ppb/ppm
    
    ##--NOAA I-CIMS--## # seems like not available for all fights
    #HCOOH = dataset.data['HCOOH_NOAACIMS'] # ppt
    #N2O5 = dataset.data['N2O5_ppt_NOAACIMS'] # ppt
    
    ##--NCAR chemiluminescence instrument--##
    O3 = dataset.data['O3_CL'] # ppbv
    NO = dataset.data['NO_CL'] # ppbv
    NO2 = dataset.data['NO2_CL'] # ppbv
    NOy = dataset.data['NOy_CL'] # ppbv
    
    ##--Nucleation mode--##
    nucleating = dataset.data['N_nucl_AMP'] # num/cm^3 STP (2.7-12 nm)
    

    
    ##--ERROR: Referenced Brock et al 2019--##
    nuc_random_err = 0.13 # +- percent value
    nucleating_random = nuc_random_err * nucleating #series
    nuc_sys_error = 0.20 # +- percent
    nucleating_systematic = nuc_sys_error * nucleating
    
    ##--Combine random and systematic uncertainties in quadrature--##
    nuc_err = 3* np.sqrt((nucleating_random)**2 + (nucleating_systematic)**2)

    ##--Aitken mode--##
    aitken = dataset.data['N_aitken_AMP'] # num/cm^3 STP (12-60 nm)
    
    aitken_random_err = 0.06
    aitken_random = aitken_random_err * aitken
    aitken_sys_err = 0.10
    aitken_systematic = aitken_sys_err * aitken
    
    aitken_err = 3* np.sqrt((aitken_random)**2 + (aitken_systematic)**2)

    
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
    
    ##########################
    ##--Place in dataframe--##
    ##########################
    
    nuc_df = pd.DataFrame({'PTemp': potential_temp, 'latitude': latitude, 
                       'nuc': nucleating, 'uncertainty': nuc_err, 
                       'O3': O3, 'OH':OH, 'HO2': HO2, 'H2O2': H2O2, 
                       'CO': CO, 'CO2': CO2, 'CO_CO2': CO_CO2, 
                       'CH4': CH4, 'NOy': NOy, 'SO2': SO2, 
                       'NO': NO, 'NO2': NO2, 'HNO3': HNO3})
    
    ##--Drop rows where the potential temperature is above 310 K for comparison to other campaigns--##
    nuc_df = nuc_df[nuc_df['PTemp']<310]
    
    ##--And where the latitude is outside of the Arctic--##
    nuc_df = nuc_df[nuc_df['latitude']>66.5]
    
    ##--Finally mask any negative values--##
    nuc_df = nuc_df.mask(nuc_df < 0)
    
    nuc_dfs.append(nuc_df)
    
    aitken_df = pd.DataFrame({'PTemp': potential_temp, 'latitude': latitude, 
                       'aitken': aitken, 'uncertainty': aitken_err, 
                       'O3': O3, 'OH':OH, 'HO2':HO2, 'H2O2': H2O2, 
                       'CO': CO, 'CO2': CO2, 'CO_CO2': CO_CO2, 
                       'CH4': CH4, 'NOy': NOy, 'SO2': SO2, 
                       'NO': NO, 'NO2': NO2, 'HNO3': HNO3})
    
    ##--Drop rows where the potential temperature is above 310 K for comparison to other campaigns--##
    aitken_df = aitken_df[aitken_df['PTemp']<310]
    
    ##--And where the latitude is outside of the Arctic--##
    aitken_df = aitken_df[aitken_df['latitude']>66.5]
    
    ##--Mask negatives--##
    aitken_df = aitken_df.mask(aitken_df < 0)
    
    aitken_dfs.append(aitken_df)

gas_var_list = ['H2O2', 'O3', 'CO', 'CO2', 'CO_CO2', 'CH4', 
                'OH', 'HO2', 'NOy', 'NO', 'NO2', 'SO2', 'HNO3']

##--Define a function that separates the top data from the rest--##
def sig_sort(df, var, particles, percentile=90):

    # Remove rows with NaNs in either variable
    temp = df[[particles, var]].dropna()

    # Threshold corresponding to the top 10%
    cutoff = temp[particles].quantile(percentile/100)

    top = temp.loc[temp[particles] >= cutoff, var]
    rest = temp.loc[temp[particles] < cutoff, var]

    violin_df = pd.DataFrame({
        'value': pd.concat([top, rest], ignore_index=True),
        'group': (['Top 10%'] * len(top)) +
                 (['Lower 90%'] * len(rest))})

    return violin_df, len(top), len(rest)
     
################
##--Plotting--##
################

gas_units = {
    'CO2': 'ppm',
    'O3': 'ppb',
    'CO': 'ppb',
    'CH4': 'ppb',
    'NOy': 'ppb',
    'NO': 'ppb',
    'NO2': 'ppb',
    'H2O2': 'ppt',
    'HO2': 'ppt',
    'HNO3': 'ppt',
    'SO2': 'ppt',
    'CO_CO2': 'ppb/ppm',
    'OH': 'ppt'}

##--Some plots will need split axes, here's where to split on the y-axis--##
split_axes = {
    'HNO3': 150,
    'SO2': 175,
    'NO': 0.3,
    'NOy': 1,
    'O3': 100}

palette_nuc = {'Top 10%': '#219ebc', 'Lower 90%': '#023047'}

palette_aitken = {'Top 10%': '#fb8500', 'Lower 90%': '#bc6c25'}

def violin_plot(df, var, particles,
                mode_name,
                palette,
                percentile=90):

    violin_df, n_top, n_rest = sig_sort(df, var, particles, percentile)

    broken_axis = var in split_axes

    ymin = violin_df['value'].min()
    ymax = violin_df['value'].max()
    pad = 0.05 * (ymax - ymin)

    if broken_axis:

        split = split_axes[var]

        fig, (ax_top, ax_bottom) = plt.subplots(
            2, 1,
            figsize=(6, 8),
            sharex=True,
            height_ratios=[1, 8],
            gridspec_kw={'hspace': 0.08}
        )

        for axis in (ax_top, ax_bottom):

            sns.violinplot(
                data=violin_df,
                x='group',
                y='value',
                hue='group',
                order=['Top 10%', 'Lower 90%'],
                palette=palette,
                cut=0,
                inner_kws={'whis_width': 0},
                legend=False,
                ax=axis
            )

        ax_bottom.set_ylim(ymin - pad, split)
        ax_top.set_ylim(split, ymax)

        # Remove duplicate labels
        ax_top.set_ylabel("")
        ax_top.set_xlabel("")
        ax_top.tick_params(labelbottom=False)

        # Use bottom axis for all remaining formatting
        main_ax = ax_bottom

    else:

        fig, main_ax = plt.subplots(figsize=(6, 5))

        sns.violinplot(
            data=violin_df,
            x='group',
            y='value',
            hue='group',
            order=['Top 10%', 'Lower 90%'],
            palette=palette,
            cut=0,
            inner_kws={'whis_width': 0},
            legend=False,
            ax=main_ax
        )

        main_ax.set_ylim(ymin - pad, ymax)

    # ----------------------------
    # Labels
    # ----------------------------

    unit = gas_units[var]

    if unit == "":
        ylabel = var
    else:
        ylabel = f"{var} ({unit})"

    main_ax.set_ylabel(ylabel, fontsize=13)
    main_ax.set_xlabel("")

    # ----------------------------
    # Sample sizes
    # ----------------------------

    main_ax.text(
        0.23,
        0.02,
        f"N={n_top}",
        transform=main_ax.transAxes,
        ha='center'
    )

    main_ax.text(
        0.77,
        0.02,
        f"N={n_rest}",
        transform=main_ax.transAxes,
        ha='center'
    )

    plt.suptitle(
        f"{mode_name} mode",
        fontsize=15,
        y=0.995
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
    
nuc_all = pd.concat(nuc_dfs, ignore_index=True)

for gas in gas_var_list:

    violin_plot(
        nuc_all,
        gas,
        particles='nuc',
        mode_name='Nucleation (2.9–12 nm)',
        palette=palette_nuc
    )
    
aitken_all = pd.concat(aitken_dfs, ignore_index=True)

for gas in gas_var_list:

    violin_plot(
        aitken_all,
        gas,
        particles='aitken',
        mode_name='Aitken (12–60 nm)',
        palette=palette_aitken
    )

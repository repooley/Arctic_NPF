# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 10:46:20 2024

@author: repooley
"""

import icartt
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from pathlib import Path

###########################
##--Establish directory--##
###########################

##--Path to this script--##
script_path = Path(__file__).resolve()

##--Path to the root which is 4 levels up in the directory--##
root = script_path.parents[4]

##--Path to raw NETCARE data--##
directory = root / "NETCARE2015" / "data" / "raw"

##--Path to utils folder containing alignment + calc scripts--##
sys.path.insert(0, str(root / "NETCARE2015" / "src" / "utils"))

##--Import modules from utils folder--##
from NETCARE_loader import load_flight # loads and aligns data
from Particle_bin_calculator import calc_particle_bins 

##--Pull R1 data for the two CPC instruments - zeroes not yet filtered out--##
CPC3_R1 = icartt.Dataset(directory / "CPC_R1" / "CPC3776_Polar6_20150408_R1_L2.ict")
CPC10_R1 = icartt.Dataset(directory / "CPC_R1" / "CPC3772_Polar6_20150408_R1_L2.ict")

##############################################################
##--Set up to pull data from all complete campaign flights--##
##############################################################

##--Flights to analyze - flights 1-10 (flight 1 has missing data)--##
flights_to_analyze = ["Flight2", "Flight3", "Flight4", "Flight5", "Flight6", 
                      "Flight7", "Flight8", "Flight9", "Flight10"]

##--Loop through each flight in the list--##
for flight in flights_to_analyze:
    
    data = load_flight(directory, flight)
    
    flight_date = data["flight_date"]

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
    
    ##--Pull in the df with particle bins--##
    particle_df = calc_particle_bins(data)
    
    df = particle_df['df']
    
    ###############
    ##--BINNING--##
    ###############
    
    ##--Define number of bins here--##
    num_bins = 124
    
    ##--Compute the minimum and maximum altitude, ignoring NaNs--##
    min_alt = df['Altitude'].min(skipna=True)
    max_alt = df['Altitude'].max(skipna=True)
    
    ##--Create bin edges from min_alt to max_alt--##
    bin_edges = np.linspace(min_alt, max_alt, num_bins + 1)
    
    ##--Pandas 'cut' splits altitude data into specified number of bins--##
    df['Altitude_bin'] = pd.cut(df['Altitude'], bins=bin_edges)
    
    ##--Group variables into each altitude bin--## 
    ##--Observed=false shows all bins, even empty ones--##
    binned_df = df.groupby('Altitude_bin', observed=False).agg(
        
       ##--Aggregate data by mean, min, and max--##
        Altitude_center=('Altitude', 'median'), 
        BC_center=('BC_mass', 'median'), 
        BC_min=('BC_mass', 'min'),
        BC_max=('BC_mass', 'max'),
        BC_25th=('BC_mass', lambda x: x.quantile(0.25)),
        BC_75th=('BC_mass', lambda x: x.quantile(0.75)),
        CPC10_conc_center=('CPC10_conc', 'median'), 
        CPC10_conc_min=('CPC10_conc', 'min'),
        CPC10_conc_max=('CPC10_conc', 'max'),
        CPC10_conc_25th=('CPC10_conc', lambda x: x.quantile(0.25)),
        CPC10_conc_75th=('CPC10_conc', lambda x: x.quantile(0.75)),
        CPC3_conc_center=('CPC3_conc', 'median'), 
        CPC3_conc_min=('CPC3_conc', 'min'),
        CPC3_conc_max=('CPC3_conc', 'max'),
        CPC3_conc_25th=('CPC3_conc', lambda x: x.quantile(0.25)),
        CPC3_conc_75th=('CPC3_conc', lambda x: x.quantile(0.75)),
        nuc_particles_center=('nuc_particles', 'median'), 
        nuc_particles_min=('nuc_particles', 'min'),
        nuc_particles_max=('nuc_particles', 'max'), 
        nuc_particles_25th=('nuc_particles', lambda x: x.quantile(0.25)),
        nuc_particles_75th=('nuc_particles', lambda x: x.quantile(0.75)),
        n_10_89_center=('n_10_89', 'median'), 
        n_10_89_min=('n_10_89', 'min'),
        n_10_89_max=('n_10_89', 'max'), 
        n_10_89_25th=('n_10_89', lambda x: x.quantile(0.25)),
        n_10_89_75th=('n_10_89', lambda x: x.quantile(0.75)),
        
        ##--Bin the uncertainty of nucleating particles--##
        nuc_error_center=('nuc_error_3sigma', 'median'),
        
        ##--And Aitken mode (10-85 nm) particles--##
        aitken_error_center=('aitken_error_3sigma', 'median')
        
        ##--Reset the index so Altitude_bin is just a column--##
    ).reset_index()
    
    ################
    ##--PLOTTING--##
    ################
    
    ##--Creates figure with 4 horizontally stacked subplots sharing a y-axis--##
    fig, axs = plt.subplots(1, 5, figsize=(15, 6), sharey=True)
    
    ##--First subplot: 10+ nm Particles vs Altitude--##
    
    ##--Averaged data in each bin is plotted against bin center--##
    axs[0].plot(binned_df['CPC10_conc_center'], binned_df['Altitude_center'], color='maroon')
    ##--Range is given by filling between data minimum and maximum for each bin--##
    axs[0].fill_betweenx(binned_df['Altitude_center'], binned_df['CPC10_conc_min'], 
                         binned_df['CPC10_conc_max'], color='indianred', alpha=0.25)
    axs[0].fill_betweenx(binned_df['Altitude_center'], binned_df['CPC10_conc_25th'],
                        binned_df['CPC10_conc_75th'], color='indianred', alpha=0.7)
    axs[0].set_ylabel('Altitude (m)', fontsize=12)
    axs[0].set_xlabel('Counts/cm\u00b3')
    axs[0].set_title('N \u2265 10 nm')
    #axs[0].set_xlim(-50, 1500)
    
    ##--Second subplot: 2.5+ nm Particles vs Altitude--##
    axs[1].plot(binned_df['CPC3_conc_center'], binned_df['Altitude_center'], color='saddlebrown')
    axs[1].fill_betweenx(binned_df['Altitude_center'], binned_df['CPC3_conc_min'], 
                         binned_df['CPC3_conc_max'], color='sandybrown', alpha=0.25)
    axs[1].fill_betweenx(binned_df['Altitude_center'], binned_df['CPC3_conc_25th'],
                        binned_df['CPC3_conc_75th'], color='sandybrown', alpha=1)
    axs[1].set_title('N \u2265 2.5 nm')
    axs[1].set_xlabel('Counts/cm\u00b3')
    #axs[1].set_xlim(-50, 2000)
    
    ##--Third subplot: Nuc particles vs Altitude--##
    axs[2].plot(binned_df['nuc_particles_center'], binned_df['Altitude_center'], color='darkslategray')
    axs[2].fill_betweenx(binned_df['Altitude_center'], binned_df['nuc_particles_min'], 
                         binned_df['nuc_particles_max'], color='cadetblue', alpha=0.25)
    axs[2].fill_betweenx(binned_df['Altitude_center'], binned_df['nuc_particles_25th'],
                        binned_df['nuc_particles_75th'], color='cadetblue', alpha=1)
    
    ##--Plot uncertainty as its own trace--##
    axs[2].plot(binned_df['nuc_error_center'], binned_df['Altitude_center'], color='crimson', 
                linestyle='dashed', label='3$\sigma$ \nuncertainty')
    
    axs[2].legend(loc='lower right')
    
    ##--Subscript 3-10--##
    axs[2].set_title('$N_{2.5-10}$')
    axs[2].set_xlabel('Counts/cm\u00b3')
    #axs[2].set_xlim(-50, 2000)
    
    ##--Fourth subplot: 10-89 nm particles vs Altitude--##
    axs[3].plot(binned_df['n_10_89_center'], binned_df['Altitude_center'], color='darkcyan')
    axs[3].fill_betweenx(binned_df['Altitude_center'], binned_df['n_10_89_min'], 
                         binned_df['n_10_89_max'], color='turquoise', alpha=0.25)
    axs[3].fill_betweenx(binned_df['Altitude_center'], binned_df['n_10_89_25th'],
                        binned_df['n_10_89_75th'], color='mediumturquoise', alpha=1)
    
    ##--Plot uncertainty as its own trace--##
    axs[3].plot(binned_df['aitken_error_center'], binned_df['Altitude_center'], color='crimson', 
                linestyle='dashed', label='3$\sigma$ \nuncertainty')
    
    axs[3].legend(loc='lower right')
    
    ##--Subscript 10-89--##
    axs[3].set_title('$N_{10-89}$')
    axs[3].set_xlabel('Counts/cm\u00b3')
    #axs[3].set_xlim(-50, 2000)
    
    ##--Fifth subplot: rBC counts--##
    axs[4].plot(binned_df['BC_center'], binned_df['Altitude_center'], color='steelblue')
    axs[4].fill_betweenx(binned_df['Altitude_center'], binned_df['BC_min'], 
                         binned_df['BC_max'], color='skyblue', alpha=0.3)
    axs[4].fill_betweenx(binned_df['Altitude_center'], binned_df['BC_25th'],
                        binned_df['BC_75th'], color='skyblue', alpha=1)
    
    axs[4].set_title('rBC Mass')
    axs[4].set_xlabel('ng/m\u00b3')
    #axs[4].set_xlim(-50, 2000)
    
    ##--Use f-string to embed flight # variable in plot title--##
    plt.suptitle(f"Vertical Particle Count Profiles - {flight.replace('Flight', 'Flight ')} ({flight_date})", fontsize=16)
    
    ##--Adjusts layout to prevent overlapping--## 
    plt.tight_layout(rect=[0, -0.02, 1, 0.99])
    
    ####################
    ##--Save figures--##
    ####################
    
    ##--Path to the directory for processed data--##
    processed_dir = root / "NETCARE2015" / "data" / "processed" 
    
    ##--Establish path to specific folder within the output directory--##
    folder_path = os.path.join(processed_dir, "Vertical Plots", "Altitude", f"{flight}")
    
    ##--Create the folder if it doesn't already exist--##
    os.makedirs(folder_path, exist_ok=True)
    
    ##--Use f-string to save file with flight# appended--##
    output_path = f"{folder_path}\\ParticleCounts_{flight}_{flight_date}"
    
    ##--Save the figure--##
    plt.savefig(output_path, dpi=300, bbox_inches='tight') 
    
    ##--Make sure plot displays in console--##
    plt.show()

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 16:05:39 2026

@author: repooley
"""

import icartt
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from scipy.stats import binned_statistic_2d
from datetime import date

###################
##--User inputs--##
###################

##--Set the base directory to project folder--##
directory = r"C:\Users\repooley\REP_PhD\Arctic_NPF\ATom2018\data"

##--The flight10 file also includes flight11--##
flights_to_analyze = ["Flight2", "Flight10", "Flight11", "Flight12"]

##--Set binning for PTemp and Latitude--##
num_bins_lat = 8
num_bins_ptemp = 8

##--Define function that creates datasets from filenames--##
def find_files(directory, flight):
    ##--flight data are stored in a folder called "raw"--##
    flight_dir = os.path.join(directory, "raw", flight)
    search_pattern = os.path.join(flight_dir, "*.ict")
    return sorted(glob.glob(search_pattern))


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
    
    ##--AIMMS Data--##
    altitude = dataset.data['G_ALT'] # in m (not sure if this is best one)
    temperature = dataset.data['T'] # in K
    pressure = dataset.data['P'] * 100 # in Pa
    RH = dataset.data['Relative_Humidity'] # wrt water, percent
    time =dataset.data['UTC_Start'] # seconds since midnight UTC
    nucleating = dataset.data['N_nucl_AMP'] # num/cm^3 STP (2.7-12 nm)
    aitken = dataset.data['N_aitken_AMP'] # num/cm^3 STP (12-60 nm)
    latitude = dataset.data['G_LAT'] # deg
    rBC = dataset.data['BC_mass_90_550_nm'] # ng/m^3 (std)
    #CO = dataset.data['CO.X'] # ppb
    
    ##--There are notable outliers in the nucleating data--##
    
    ##--First convert to a series for calc--##
    nucleating_series = pd.Series(nucleating)
    
    ##--REMOVE OUTLIERS above 99th percentile--##
    p = 0.99
    
    ##--Compute threshold for each UHSAS column--##
    nucleating_thresh = nucleating_series.quantile(p)
    
    ##--keep only rows where each bin is below its threshold--##
    nucleating_filtered = nucleating_series[nucleating_series.le(nucleating_thresh)]
    
    nucleating_filtered = nucleating_series.mask(
    nucleating_series <= nucleating_thresh)
 
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
    ##--Create 2D histogram--##
    ###########################
    
    ##--2.5-10nm, 'nucleating'--##
    nuc_df = pd.DataFrame({'PTemp': potential_temp, 'Latitude': latitude, 'nuc_particles': nucleating})
    ##--Drop rows where the potential temperature is above 310 K for comparison to other campaigns--##
    nuc_df = nuc_df[nuc_df['PTemp']<310]
    nuc_clean_df = nuc_df.dropna()
    
    ##--10-89nm, 'growth'--##
    grow_df = pd.DataFrame({'PTemp': potential_temp, 'Latitude': latitude, 'grow_particles': aitken})
    ##--Drop rows where the potential temperature is above 310 K for comparison to other campaigns--##
    grow_df = grow_df[grow_df['PTemp']<310]
    grow_clean_df = grow_df.dropna()
    
    ##--rBC--##
    rBC_df = pd.DataFrame({'PTemp': potential_temp, 'Latitude': latitude, 'rBC': rBC})
    ##--Drop rows where the potential temperature is above 310 K for comparison to other campaigns--##
    rBC_df = rBC_df[rBC_df['PTemp']<310]
    rBC_clean_df = rBC_df.dropna()
    
    ##--Place lat/ptemp in a dataframe to clean before computing bin edges--##
    lat_ptemp_df = pd.DataFrame({'latitude': latitude, 'PTemp': potential_temp})
    ##--Drop PTemps above 310 K to directly compare to other campaigns--##
    lat_ptemp_df = lat_ptemp_df[lat_ptemp_df['PTemp']<310]
    ##--Constrain measurement region to the Arctic--##
    lat_ptemp_df = lat_ptemp_df[lat_ptemp_df['latitude']>66.5]
    
    ##--Compute global min/max values across all data BEFORE dropping NaNs--##
    lat_min, lat_max = np.nanmin(lat_ptemp_df['latitude']), np.nanmax(lat_ptemp_df['latitude'])
    ptemp_min, ptemp_max = np.nanmin(lat_ptemp_df['PTemp']), np.nanmax(lat_ptemp_df['PTemp'])
    
    ##--Generate common bin edges using specified number of bins--##
    common_lat_bin_edges = np.linspace(lat_min, lat_max, num_bins_lat + 1)
    common_ptemp_bin_edges = np.linspace(ptemp_min, ptemp_max, num_bins_ptemp + 1)
    
    ##--N(2.5-10)--##
    nuc_bin_medians, _, _, _ = binned_statistic_2d(nuc_clean_df['Latitude'], 
        nuc_clean_df['PTemp'], nuc_clean_df['nuc_particles'], statistic='median', bins=[common_lat_bin_edges, common_ptemp_bin_edges])
    
    ##--N(10-89)--##
    grow_bin_medians, _, _, _ = binned_statistic_2d(grow_clean_df['Latitude'], 
        grow_clean_df['PTemp'], grow_clean_df['grow_particles'], statistic='median', bins=[common_lat_bin_edges, common_ptemp_bin_edges])
    
    ##--rBC--##
    rBC_bin_medians, _, _, _ = binned_statistic_2d(rBC_clean_df['Latitude'], 
       rBC_clean_df['PTemp'], rBC_clean_df['rBC'], statistic='median', bins=[common_lat_bin_edges, common_ptemp_bin_edges])
    
    ################
    ##--PLOTTING--##
    ################
    
    ##--Particles larger than 3 nm--##
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    
    ##--Make special color map where 0 values are white--##
    new_cmap = plt.get_cmap('viridis')
    ##--Values under specified minimum will be white--##
    new_cmap.set_under('w')
    
    ##--Use pcolormesh for the plot, set minimum value for viridis colors as 1--##
    nuc_plot = ax1.pcolormesh(common_lat_bin_edges, common_ptemp_bin_edges, nuc_bin_medians.T,  # Transpose to align correctly
        shading='auto', cmap=new_cmap, vmin=0, vmax=2500)
    
    ##--Add colorbar--##
    cb = fig1.colorbar(nuc_plot, ax=ax1)
    cb.minorticks_on()
    cb.ax.tick_params(labelsize=16)
    cb.set_label('Particles 2.7-12 nm $(Counts/cm^{3})$', fontsize=16)
    
    ##--Set axis labels--##
    ax1.set_xlabel('Latitude (°)', fontsize=16)
    ax1.set_ylabel('Potential Temperature \u0398 (K)', fontsize=16)
    ax1.tick_params(axis='both', labelsize=16)
    ax1.set_title(f"2.7-12 nm Particles - ATom 4 {flight.replace('Flight', 'Flight ')} ({flight_date})", fontsize=18, pad=20)
    #ax1.set_ylim(238, 301)
    #ax1.set_xlim(79.5, 83.7)
    
    ##--Use f-string to save file with flight# appended--##
    #CPC3_output_path = f"{output_path}\\CPC3/PTempLatitude/{flight}"
    #plt.savefig(CPC3_output_path, dpi=600, bbox_inches='tight') 
    
    plt.tight_layout()
    plt.show()
    
    ##--Particles larger than 10 nm--##
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    
    ##--Use pcolormesh for the plot, set minimum for viridis colors as 1--##
    grow_plot = ax2.pcolormesh(common_lat_bin_edges, common_ptemp_bin_edges, grow_bin_medians.T,  # Transpose to align correctly
        shading='auto', cmap=new_cmap, vmin=0, vmax=2000)
    
    ##--Add colorbar--##
    cb2 = fig2.colorbar(grow_plot, ax=ax2)
    cb2.minorticks_on()
    cb2.ax.tick_params(labelsize=16)
    cb2.set_label('Particles >10 nm $(Counts/cm^{3})$', fontsize=16)
    
    ##--Set axis labels--##
    ax2.set_xlabel('Latitude (°)', fontsize=16)
    ax2.set_ylabel('Potential Temperature \u0398 (K)', fontsize=16)
    ax2.tick_params(axis='both', labelsize=16)
    ax2.set_title(f"12-60 nm Particles - ATom 4 {flight.replace('Flight', 'Flight ')} ({flight_date})", fontsize=18, pad=20)
    #ax2.set_ylim(238, 301)
    #ax2.set_xlim(79.5, 83.7)
    
    ##--Use f-string to save file with flight# appended--##
    #CPC10_output_path = f"{output_path}\\CPC10/PTempLatitude/{flight}"
    #plt.savefig(CPC10_output_path, dpi=600, bbox_inches='tight') 
    
    plt.tight_layout()
    plt.show()
    
    
    ##--rBC--##
    fig3, ax3 = plt.subplots(figsize=(8,6))
    
    rBC_plot = ax3.pcolormesh(common_lat_bin_edges, common_ptemp_bin_edges, rBC_bin_medians.T,
              shading='auto', cmap=new_cmap)
    
    ##--Add colorbar--##
    cb3 = fig3.colorbar(rBC_plot, ax=ax3)
    cb3.minorticks_on()
    cb3.ax.tick_params(labelsize=16)
    cb2.set_label('rBC $(ng/m^{3})$', fontsize=16)
    
    ##--Set axis labels--##
    ax3.set_xlabel('Latitude (°)', fontsize=16)
    ax3.set_ylabel('Potential Temperature \u0398 (K)', fontsize=16)
    ax3.tick_params(axis='both', labelsize=16)
    ax3.set_title(f"rBC - ATom 4 {flight.replace('Flight', 'Flight ')} ({flight_date})", fontsize=18, pad=20)
    #ax3.set_ylim(238, 301)
    #ax3.set_xlim(79.5, 83.7)
    
    '''
    ########################
    ##--Diagnostic Plots--##
    ########################
    
    
    ##--Counts per bin for CPC3 data--##
    CPC3_bin_counts, _, _, _ = binned_statistic_2d(CPC3_clean_df['Latitude'], 
        CPC3_clean_df['PTemp'], CPC3_clean_df['CPC3'], statistic='count', bins=[common_lat_bin_edges, common_ptemp_bin_edges])
     
    ##--Counts per bin for CPC10 data--##
    CPC10_bin_counts, _, _, _ = binned_statistic_2d(CPC10_clean_df['Latitude'], 
        CPC10_clean_df['PTemp'], CPC10_clean_df['CPC10'], statistic='count', bins=[common_lat_bin_edges, common_ptemp_bin_edges])
    
    ##--Counts per bin for N3-10 particles--##
    nuc_bin_counts, _, _, _ = binned_statistic_2d(nuc_clean_df['Latitude'], 
        nuc_clean_df['PTemp'], nuc_clean_df['nuc_particles'], statistic='count', bins=[common_lat_bin_edges, common_ptemp_bin_edges])
    
    ##--Counts per bin for N10-89 particles--##
    grow_bin_counts, _, _, _ = binned_statistic_2d(grow_clean_df['Latitude'], 
        grow_clean_df['PTemp'], grow_clean_df['n_10_89'], statistic='count', bins=[common_lat_bin_edges, common_ptemp_bin_edges])
    
    ##--Plotting--##
    
    ##--Particles larger than 3 nm--##
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    
    ##--Make special color map where 0 values are white--##
    new_cmap = plt.get_cmap('inferno')
    ##--Values under specified minimum will be white--##
    new_cmap.set_under('w')
    
    ##--Use pcolormesh for the plot, set minimum value for viridis colors as 1--##
    CPC3_plot = ax1.pcolormesh(common_lat_bin_edges, common_ptemp_bin_edges, CPC3_bin_counts.T,  # Transpose to align correctly
        shading='auto', cmap=new_cmap, vmin=1, vmax=1250)
    
    ##--Add colorbar--##
    cb = fig1.colorbar(CPC3_plot, ax=ax1)
    cb.minorticks_on()
    cb.ax.tick_params(labelsize=16)
    cb.set_label('Number of Data Points', fontsize=16)
    
    # Set axis labels
    ax1.set_xlabel('Latitude (°)', fontsize=16)
    ax1.set_ylabel('Potential Temperature \u0398 (K)', fontsize=16)
    ax1.tick_params(axis='both', labelsize=16)
    ax1.set_title(f"Particles >2.5 nm Counts per Bin - {flight.replace('Flight', 'Flight ')}", fontsize=18)
    #ax1.set_ylim(238, 301)
    #ax1.set_xlim(79.5, 83.7)
    
    ##--Use f-string to save file with flight# appended--##
    CPC3_diag_output_path = f"{output_path}\\CPC3/PTempLatitude/{flight}_diagnostic"
    plt.savefig(CPC3_diag_output_path, dpi=600, bbox_inches='tight') 
    
    plt.tight_layout()
    plt.show()
    
    ##--Particles larger than 10 nm--##
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    
    ##--Use pcolormesh for the plot, set minimum for viridis colors as 1--##
    CPC10_plot = ax2.pcolormesh(common_lat_bin_edges, common_ptemp_bin_edges, CPC10_bin_counts.T,  # Transpose to align correctly
        shading='auto', cmap=new_cmap, vmin=1, vmax=1250)
    
    ##--Add colorbar--##
    cb2 = fig2.colorbar(CPC10_plot, ax=ax2)
    cb2.minorticks_on()
    cb2.ax.tick_params(labelsize=16)
    cb2.set_label('Number of Data Points', fontsize=16)
    
    ##--Set axis labels--##
    ax2.set_xlabel('Latitude (°)', fontsize=16)
    ax2.set_ylabel('Potential Temperature \u0398 (K)', fontsize=16)
    ax2.tick_params(axis='both', labelsize=16)
    ax2.set_title(f"Particles >10 nm Counts per Bin - {flight.replace('Flight', 'Flight ')}", fontsize=18)
    #ax2.set_ylim(238, 301)
    #ax2.set_xlim(79.5, 83.7)
    
    ##--Use f-string to save file with flight# appended--##
    CPC10_diag_output_path = f"{output_path}\\CPC10/PTempLatitude/{flight}_diagnostic"
    plt.savefig(CPC10_diag_output_path, dpi=600, bbox_inches='tight') 
    
    plt.tight_layout()
    plt.show()
    
    ##--Nucleating particles--##
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    
    ##--Use pcolormesh for the plot and use viridis for values greater than 1--##
    nuc_plot = ax3.pcolormesh(common_lat_bin_edges, common_ptemp_bin_edges, nuc_bin_counts.T,  # Transpose to align correctly
        shading='auto', cmap=new_cmap, vmin=1, vmax=1000)
    
    ##--Add colorbar--##
    cb3 = fig3.colorbar(nuc_plot, ax=ax3)
    cb3.minorticks_on()
    cb3.ax.tick_params(labelsize=16)
    cb3.set_label('Number of Data Points', fontsize=16)
    
    ##--Set axis labels--##
    ax3.set_xlabel('Latitude (°)', fontsize=16)
    ax3.set_ylabel('Potential Temperature \u0398 (K)', fontsize=16)
    ax3.tick_params(axis='both', labelsize=16)
    ax3.set_title(f"2.5-10 nm Particle Counts per Bin - {flight.replace('Flight', 'Flight ')}", fontsize=18)
    #ax3.set_ylim(238, 301)
    #ax3.set_xlim(79.5, 83.7)
    
    ##--Use f-string to save file with flight# appended--##
    nuc_diag_output_path = f"{output_path}\\Nucleating/PTempLatitude/{flight}_diagnostic"
    plt.savefig(nuc_diag_output_path, dpi=600, bbox_inches='tight') 
    
    ##--10-89 nm particles--##
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    
    ##--Use pcolormesh for the plot and use viridis for values greater than 1--##
    grow_diag_plot = ax4.pcolormesh(common_lat_bin_edges, common_ptemp_bin_edges, grow_bin_counts.T,  # Transpose to align correctly
        shading='auto', cmap=new_cmap, vmin=1, vmax=1000)
    
    ##--Add colorbar--##
    cb4 = fig4.colorbar(grow_diag_plot, ax=ax4)
    cb4.minorticks_on()
    cb4.ax.tick_params(labelsize=16)
    cb4.set_label('Number of Data Points', fontsize=16)
    
    ##--Set axis labels--##
    ax4.set_xlabel('Latitude (°)', fontsize=16)
    ax4.set_ylabel('Potential Temperature \u0398 (K)', fontsize=16)
    ax4.tick_params(axis='both', labelsize=16)
    ax4.set_title(f"2.5-10 nm Particle Counts per Bin - {flight.replace('Flight', 'Flight ')}", fontsize=18)
    #ax4.set_ylim(238, 301)
    #ax4.set_xlim(79.5, 83.7)
    
    ##--Use f-string to save file with flight# appended--##
    grow_diag_output_path = f"{output_path}\\Nucleating/PTempLatitude/{flight}_diagnostic"
    plt.savefig(grow_diag_output_path, dpi=600, bbox_inches='tight') 
    
    plt.tight_layout()
    plt.show()
    '''

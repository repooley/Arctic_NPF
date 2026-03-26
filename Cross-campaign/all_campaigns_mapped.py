# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 10:58:01 2025

@author: repooley
"""

import icartt
import os
import glob
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from pathlib import Path
import sys

#########################
##--Open ICARTT Files--##
#########################

##--Path to this script--##
script_path = Path(__file__).resolve()

##--Path to the root which is 1 level up in the directory--##
root = script_path.parents[1]
 
##--Base directories--##
NETCARE = r"C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data\raw"
FIREACE = r"C:\Users\repooley\REP_PhD\Arctic_NPF\FIREACE1998\data\raw"
PAMARCMiP = r"C:\Users\repooley\REP_PhD\Arctic_NPF\PAMARCMiP2012\data\raw"
ATom = r"C:\Users\repooley\REP_PhD\Arctic_NPF\ATom2018\data\raw"
 
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
 
##--Choose which flights to analyze here!--##

##--ATom4: only four flights transect the Arctic region--##
ATom_to_analyze = ["Flight2", "Flight10", "Flight11", "Flight12"]

##--NETCARE: excluding Flights 1 (no UHSAS) and 4 (bad CPC data)--##
NETCARE_to_analyze = ["Flight2", "Flight3", "Flight5", 
    "Flight6", "Flight7", "Flight8", "Flight9", "Flight10"]

PAMARCMiP_to_analyze = ["Flight1", "Flight2", "Flight3", "Flight4",
    "Flight5", "Flight6", "Flight7", "Flight8", "Flight9"]

##--FIRE-ACE: missing data for flights 1, 2, 4, 5, 6--##
FIREACE_to_analyze = ["Flight3", "Flight7", "Flight8", "Flight9", "Flight10", 
    "Flight11", "Flight12", "Flight13", "Flight14", "Flight15", "Flight16", 
    "Flight17", "Flight18"]

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
    
####################
##--NETCARE data--##
####################

##--Store processed data here: --##
NETCARE_dfs = []

##--Use NETCARE Utils files to align data and calculate nucleation mode--##
##--Path to utils folder containing alignment + calc scripts--##
sys.path.insert(0, str(root / "NETCARE2015" / "src" / "utils"))

##--Import modules from utils folder--##
from NETCARE_loader import load_flight # loads and aligns data
from Particle_bin_calculator import calc_particle_bins 

##--Set NETCARE directory--##
##--Path to raw NETCARE data--##
NETCARE_directory = root / "NETCARE2015" / "data" / "raw"
 
##--Loop through each flight, pulling and analyzing data--##
for flight in NETCARE_to_analyze:
    ##--Follow which flight is processing--##
    print(f"Processing NETCARE {flight}...")
    
    NETCARE_data = load_flight(NETCARE_directory, flight)
    
    ##--AIMMS data--##
    AIMMS = NETCARE_data["AIMMS"]

    NETCARE_altitude = AIMMS.data["Alt"] # m
    NETCARE_latitude = AIMMS.data["Lat"] # deg N
    NETCARE_longitude = AIMMS.data['Lon'] # deg
    NETCARE_temperature = AIMMS.data["Temp"] + 273.15 # K
    NETCARE_pressure = AIMMS.data["BP"] # pa
    
    ##--Calculate PTemp--##
    NETCARE_ptemp = calc_ptemp(NETCARE_temperature, NETCARE_pressure)
    
    ##--Complete particle calculations using imported function--##
    NETCARE_particle_output = calc_particle_bins(NETCARE_data)
    
    ##--Pull the resulting df containing particle data--##
    NETCARE_particle_df = NETCARE_particle_output['df']
    
    ##--Pull nucleation mode data from particle df--##
    NETCARE_nucleation = NETCARE_particle_df['nuc_particles']
    
    ##--Constrain nucleation mode to significant data, above 134 counts/cm^3--##
    NETCARE_nucleation[NETCARE_nucleation < 134] = np.nan
    
    NETCARE_df = pd.DataFrame({'Lat': NETCARE_latitude,'Lon': NETCARE_longitude, 'Alt': NETCARE_altitude, 
                               'PTemp': NETCARE_ptemp, 'nuc': NETCARE_nucleation})
    NETCARE_dfs.append(NETCARE_df[['Lat', 'Lon', 'Alt', 'PTemp', 'nuc']])
    
##--Repeat for FIRE-ACE 1998--##
FIREACE_dfs = []

for flight in FIREACE_to_analyze:  
    print(f"Processing FIREACE {flight}...")    
    
    FIREACE_dir = os.path.join(FIREACE, flight)
    
    data_files = find_files(FIREACE_dir, "FIREACE")
    if data_files:
        FIREACE_data = pd.read_csv(data_files[0])
    else:
        print(f"No file found for {flight}. Skipping...")
        continue  
        
    #################
    ##--Pull data--##
    #################

    FIREACE_altitude = FIREACE_data['Altitude'] # in m
    FIREACE_latitude = FIREACE_data['Latitude'] # in degrees
    FIREACE_longitude = FIREACE_data['Longitude'] # in degrees
    FIREACE_pressure = FIREACE_data['Pressure'] * 100 # in Pa
    FIREACE_temperature = FIREACE_data['Temperature'] + 273.15 # in K
    FIREACE_RH = FIREACE_data['RH'] # percent wrt water
    
    ##--Particle data, 3 and 10 nm cutoffs, respectively--##
    FIREACE_CPC3_data = FIREACE_data['CN3025_corrected'] # Uncorrected data has a flow issue
    FIREACE_CPC10_data = FIREACE_data['CN7610']
    
    ######################
    ##--Calc N(2.5-10)--##
    ######################
    
    ##--Convert to STP!--##
    P_STP = 101325  # Pa
    T_STP = 273.15  # K
    
    ##--Create empty list for CPC3 particles--##
    CPC3_conc_STP = []
    
    for CPC3, T, P in zip(FIREACE_CPC3_data, FIREACE_temperature, FIREACE_pressure):
        if np.isnan(CPC3) or np.isnan(T) or np.isnan(P):
            ##--Append with NaN if any input is NaN--##
            CPC3_conc_STP.append(np.nan)
        else:
            ##--Perform conversion if all inputs are valid--##
            CPC3_conversion = CPC3 * (P_STP / P) * (T / T_STP)
            CPC3_conc_STP.append(CPC3_conversion)
        
    ##--Create empty list for CPC10 particles--##
    CPC10_conc_STP = []
    
    for CPC10, T, P in zip(FIREACE_CPC10_data, FIREACE_temperature, FIREACE_pressure):
        if np.isnan(CPC10) or np.isnan(T) or np.isnan(P):
            ##--Append with NaN if any input is NaN--##
            CPC10_conc_STP.append(np.nan)
        else:
            ##--Perform conversion if all inputs are valid--##
            CPC10_conversion = CPC10 * (P_STP / P) * (T / T_STP)
            CPC10_conc_STP.append(CPC10_conversion)
    
    ##--Creates a Pandas dataframe for particle data--##
    CPC_df = pd.DataFrame({'Altitude': FIREACE_altitude, 'CPC3_conc':CPC3_conc_STP, 'CPC10_conc': CPC10_conc_STP})
    
    ##--Calculate N3-10 particles--##
    nuc_particles = (CPC_df['CPC3_conc'] - CPC_df['CPC10_conc'])
    
    ##--Change calculated particle counts less than zero to NaN--##
    FIREACE_nuc_particles = np.where(nuc_particles >= 0, nuc_particles, np.nan)
    
    ##--Constrain nucleation mode to significant data, above 134 counts/cm^3--##
    FIREACE_nuc_particles[FIREACE_nuc_particles < 134] = np.nan
    
    ##--Calculate potential temperature--##
    FIREACE_ptemp = calc_ptemp(FIREACE_temperature, FIREACE_pressure)
    
    ##--Create df--##
    FIREACE_df = pd.DataFrame({'Lat': FIREACE_latitude,'Lon': FIREACE_longitude, 
                               'Alt': FIREACE_altitude, 'nuc':FIREACE_nuc_particles, 
                               'PTemp': FIREACE_ptemp})
    
    ##--FIRE-ACE longitudes are all west of meridian, but denoted as positive--##
    FIREACE_df['Lon'] = FIREACE_df['Lon'] * -1
    
    ##--Append df to list outside of loop--##
    FIREACE_dfs.append(FIREACE_df[['Lat', 'Lon', 'Alt', 'nuc', 'PTemp']])
    
##--Repeat for PAMARCMiP 2012--##
PAMARCMiP_dfs = []

for flight in PAMARCMiP_to_analyze:  
    print(f"Processing PAMARCMiP {flight}...")    
    
    PAMARCMiP_dir = os.path.join(PAMARCMiP, flight)
    
    PAMARCMiP_files = find_files(PAMARCMiP_dir, "PA")
    if PAMARCMiP_files:
        PAMARCMiP_data = pd.read_csv(PAMARCMiP_files[0])
    else:
        print(f"No file found for {flight}. Skipping...")
        continue  
        
    #################
    ##--Pull data--##
    #################
    
    PAMARCMiP_altitude = PAMARCMiP_data['Altitude'] # in m
    PAMARCMiP_latitude = PAMARCMiP_data['Latitude'] # in degrees
    PAMARCMiP_longitude = PAMARCMiP_data['Longitude'] # in degrees
    PAMARCMiP_temperature = PAMARCMiP_data['Temp'] + 273.15 # in K
    PAMARCMiP_pressure = PAMARCMiP_data['Pressure'] # in pa
    
    ##--Calculate potential temperature--##
    PAMARCMiP_ptemp = calc_ptemp(PAMARCMiP_temperature, PAMARCMiP_pressure)

    ##--Populate df, note no nucleation mode for PAMARCMiP--##
    PAMARCMiP_df = pd.DataFrame({'Lat': PAMARCMiP_latitude,'Lon': PAMARCMiP_longitude, 
                                 'Alt': PAMARCMiP_altitude, 'PTemp':PAMARCMiP_ptemp})
    
    ##--Append list outside of loop--##
    PAMARCMiP_dfs.append(PAMARCMiP_df[['Lat', 'Lon', 'Alt', 'PTemp']])

##--Repeat for ATom 2018--##
ATom_dfs = []

for flight in ATom_to_analyze:  
    print(f"Processing PAMARCMiP {flight}...")    
    
    ATom_dir = os.path.join(ATom, flight)
    
    ATom_files = find_files(ATom_dir, "MER")
    if ATom_files:
        ATom_data = icartt.Dataset(ATom_files[0])
    else:
        print(f"No file found for {flight}. Skipping...")
        continue  
        
    #################
    ##--Pull data--##
    #################
    
    ##--AIMMS Data--##
    ATom_altitude = ATom_data.data['G_ALT'] # in m
    ATom_latitude = ATom_data.data['LAT_AMSSD'] # in degrees
    ATom_longitude = ATom_data.data['LON_AMSSD'] # in degrees
    ATom_nucleating = ATom_data.data['N_nucl_AMP'] # num/cm^3 STP (2.7-12 nm)
    ATom_temperature = ATom_data.data['T'] # in K
    ATom_pressure = ATom_data.data['P'] * 100 # in Pa
    
    ##--Constrain latitude to the Arctic region--##
    ATom_latitude[ATom_latitude < 66.5] = np.nan
    
    ##--Constrain nucleation mode to significant data, above 134 counts/cm^3--##
    ATom_nucleating[ATom_nucleating < 134] = np.nan
    
    ##--Calculate potential temperature--##
    ATom_ptemp = calc_ptemp(ATom_temperature, ATom_pressure)

    ##--Create df--##
    ATom_df = pd.DataFrame({'Lat': ATom_latitude,'Lon': ATom_longitude, 'Alt': ATom_altitude, 'PTemp':ATom_ptemp, 
                            'nuc':ATom_nucleating})
    
    ##--Append list outside of loop with df--##
    ATom_dfs.append(ATom_df[['Lat', 'Lon', 'Alt', 'PTemp', 'nuc']])
    
    
################################
##--Binning for alt vs ptemp--##
################################

##--Agglomerate all dataframes in each campaign list--##
ATom_all = pd.concat(ATom_dfs, ignore_index=True)
NETCARE_all = pd.concat(NETCARE_dfs, ignore_index=True)
PAMARCMiP_all = pd.concat(PAMARCMiP_dfs, ignore_index=True)
FIREACE_all = pd.concat(FIREACE_dfs, ignore_index=True)

##--Define a vertical binning function to apply to all four campaigns--##
def bin_vertical_profile(df, n_bins=64):
    
    df = df.dropna(subset=['Alt', 'PTemp'])

    ##--Bin by ALTITUDE--##
    alt_min = 0
    alt_max = 8000 # m

    bins = np.linspace(alt_min, alt_max, n_bins + 1)

    df['alt_bin'] = pd.cut(df['Alt'], bins)

    grouped = df.groupby('alt_bin')

    median_ptemp = grouped['PTemp'].median()
    alt_center = grouped['Alt'].median()

    q1 = grouped['PTemp'].quantile(0.25)
    q3 = grouped['PTemp'].quantile(0.75)

    return alt_center, median_ptemp, q1, q3

##--Apply function to all four campaigns--##
alt_ATom, ptemp_ATom, q1_ATom, q3_ATom = bin_vertical_profile(ATom_all)
alt_NETCARE, ptemp_NETCARE, q1_NETCARE, q3_NETCARE = bin_vertical_profile(NETCARE_all)
alt_PAMARCMiP, ptemp_PAMARCMiP, q1_PAMARCMiP, q3_PAMARCMiP = bin_vertical_profile(PAMARCMiP_all)
alt_FIREACE, ptemp_FIREACE, q1_FIREACE, q3_FIREACE = bin_vertical_profile(FIREACE_all)
        
################
##--Plotting--##
################

##--Create a wide plot with 4 panels--##
fig = plt.figure(figsize=(12, 10))
##--First column should be much wider than second--##
gs = gridspec.GridSpec(2, 2, width_ratios=[2.5, 1])

##--Campaign map plot
ax1 = fig.add_subplot(gs[0, 0], projection= ccrs.NorthPolarStereo(central_longitude=-90))  

##--Nucleation map plot (no 2012)--##
ax2 = fig.add_subplot(gs[1, 0], projection= ccrs.NorthPolarStereo(central_longitude=-90))  

##--Full height plot for alt vs ptemp (cover 2 panels)--##
ax3 = fig.add_subplot(gs[:, 1])  


##--Create land and ocean features to add to map subplots--##
land = cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='black', facecolor='lightgray')
ocean = cfeature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='none', facecolor='lightblue')

##--Add the features to the map subplots and also plot coastlines--##
ax1.add_feature(ocean)
ax1.add_feature(land)
ax1.coastlines()
ax2.add_feature(ocean)
ax2.add_feature(land)
ax2.coastlines()

##--Set the aspect ratio to 'auto' for ax1 & ax2 to fill whole area with the map--##
ax1.set_aspect('auto')
ax2.set_aspect('auto')

##--Set the map extent to encompass all flights--##
ax1.set_extent([-180, 0, 65, 90], crs=ccrs.PlateCarree())
ax2.set_extent([-180, 0, 65, 90], crs=ccrs.PlateCarree())

##--ATom--##
for df in ATom_dfs:
    ##--Plot flight track on first subplot--##
    ax1.scatter(df['Lon'], df['Lat'], color='purple',
               transform=ccrs.PlateCarree(), s=20, zorder=3)
    ##--Plot flight track and color by nucleation mode on second subplot--##
    ##--Ref this first nuc plot for the colorbar--##
    nuc_plot = ax2.scatter(df['Lon'], df['Lat'], c=df['nuc'], 
                cmap='viridis', transform=ccrs.PlateCarree(), s=20, zorder=3, 
                vmin=134, vmax=500)

##--Make new 'plot' on ax1 for a legend with all campaigns--##
ax1.scatter([], [], color='purple', label='ATom 2018', s=16)
   
##--Plot all NETCARE flights--##
for df in NETCARE_dfs:
    ax1.scatter(df['Lon'], df['Lat'], color='teal',
               transform=ccrs.PlateCarree(), s=20, zorder=3)
    ax2.scatter(df['Lon'], df['Lat'], c=df['nuc'], cmap='viridis', 
                transform=ccrs.PlateCarree(), s=20, zorder=3, 
                vmin=134, vmax=500)
    
##--Empty plot used to set up legend for NETCARE--##
ax1.scatter([], [], color='teal', label='NETCARE 2015', s=16)

##--PAMARCMiP--##
for df in PAMARCMiP_dfs:
    ax1.scatter(df['Lon'], df['Lat'], color='olive',
               transform=ccrs.PlateCarree(), s=20, zorder=3)
    ##--No nucleation mode to scatter on ax2.--##

ax1.scatter([], [], color='olive', label='PAMARCMiP 2012', s=16)

##--FIREACE--##
for df in FIREACE_dfs:
    ax1.scatter(df['Lon'], df['Lat'], color='goldenrod',
               transform=ccrs.PlateCarree(), s=20, zorder=3)
    ax2.scatter(df['Lon'], df['Lat'], c=df['nuc'], cmap='viridis', 
                transform=ccrs.PlateCarree(), s=20, zorder=3, 
                vmin=134, vmax=500)

ax1.scatter([], [], color='goldenrod', label='FIRE-ACE 1998', s=16)

##--Add locations with star markers and labels--##
locations = {
    "Alert, NU": (-62.34, 82.50),
    "Eureka, NU": (-85.93, 79.98),
    "Resolute, NU": (-94.8292, 74.6885),
    "Inuvik, NWT": (-133.72, 68.36),
    "Ny Alesund, NO": (11.99, 78.93),
    "Utqiagvik, AK": (-156.788605, 71.290558),
}

for name, (lon, lat) in locations.items():
    ax1.scatter(lon, lat, color='red', marker='*', s=250, edgecolor='black', transform=ccrs.PlateCarree(), zorder=4)

##--Add legend for flight numbers--##
legend = ax1.legend(loc='upper right', fontsize=12, framealpha=1, markerscale=3, ncol=1)

##--Add data to the third subplot outside of for loops established above--##
ax3.fill_betweenx(alt_ATom, q1_ATom, q3_ATom, color='purple', alpha=0.3)
ax3.plot(ptemp_ATom, alt_ATom, label='ATom 2018', c='purple')
ax3.fill_betweenx(alt_NETCARE, q1_NETCARE, q3_NETCARE, color='teal', alpha=0.3)
ax3.plot(ptemp_NETCARE, alt_NETCARE, label='NETCARE 2015', c='teal')
ax3.fill_betweenx(alt_PAMARCMiP, q1_PAMARCMiP, q3_PAMARCMiP, color='olive', alpha=0.3)
ax3.plot(ptemp_PAMARCMiP, alt_PAMARCMiP, label='PAMARCMiP 2012', c='olive')
ax3.fill_betweenx(alt_FIREACE, q1_FIREACE, q3_FIREACE, color='goldenrod', alpha=0.3)
ax3.plot(ptemp_FIREACE, alt_FIREACE, label='FIRE-ACE 1998', c='goldenrod')

ax3.set_xlabel('Median Potential Temperature (K)')
ax3.set_ylabel('Altitude (m)')
ax3.legend()

##--Add legend for nucleating data--##
cbar = fig.colorbar(nuc_plot, ax=ax2, orientation='horizontal', pad=0.02, shrink=0.8)
cbar.set_label('Nucleating Particle Concentration')

##--Add gridlines--##
gl = ax1.gridlines(draw_labels=True)
gl2 = ax2.gridlines(draw_labels=True)

plt.show()

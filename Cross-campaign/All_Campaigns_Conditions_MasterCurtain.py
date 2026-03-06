# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 07:30:12 2026

@author: repooley
"""

import icartt
import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic_2d
import matplotlib.pyplot as plt 
import cmcrameri as cm
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors

###################
##--User inputs--##
###################

##--Set the base directories to project folder--##
ATom_directory = r"C:\Users\repooley\REP_PhD\Arctic_NPF\ATom2018\data\raw"
NETCARE_directory = r"C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data\raw"
PAMARCMiP_directory = r"C:\Users\repooley\REP_PhD\Arctic_NPF\PAMARCMiP2012\data\raw"
FIREACE_directory = r"C:\Users\repooley\REP_PhD\Arctic_NPF\FIREACE1998\data\raw"

##--Choose which flights to analyze here!--##
##--ATom--##
ATom_to_analyze = ["Flight2", "Flight10", "Flight11", "Flight12"]

##--NETCARE--##
NETCARE_to_analyze = ["Flight2", "Flight3", "Flight4", "Flight5", "Flight6", 
                      'Flight7', 'Flight8', 'Flight9', 'Flight10']

##--PAMARCMiP--##
PAMARCMiP_to_analyze = ["Flight1", "Flight2", "Flight3", "Flight6", 
                      "Flight7", "Flight8", "Flight9"]

##--FIRE-ACE--##
FIREACE_to_analyze = ["Flight3",  "Flight7", "Flight8", "Flight9", "Flight10", 
                      "Flight11", "Flight12", "Flight13", "Flight14", 
                      "Flight15", "Flight16", "Flight17", "Flight18"]

##--Set number of bins for latitude and potential temperature--##
num_bins_lat = 12
num_bins_ptemp = 12

##--Base output path for figures in directory--##
#output_path = r"C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data\processed\CurtainPlots\CondensationSink"

##################
##--Open Files--##
##################

##--Define function that creates datasets from filenames--##
def find_files(directory, flight, partial_name):
    ##--flight data are stored in a folder called "raw"--##
    flight_dir = os.path.join(directory, flight)
    search_pattern = os.path.join(flight_dir, f"*{partial_name}*")
    return sorted(glob.glob(search_pattern))


##--ATom--##
ATom_conditions_dfs = []

##--Loop through each flight in the list--##
for flight in ATom_to_analyze:
    
    #########################
    ##--Open ICARTT Files--##
    #########################

    ##--Pull the merged datasets--##
    dataset = icartt.Dataset(find_files(ATom_directory, flight, "MER")[0])

    #################
    ##--Pull data--##
    #################

    altitude = dataset.data['G_ALT'] # in m (not sure if this is best one)
    latitude = dataset.data['LAT_AMSSD'] # deg
    temperature = dataset.data['T'] # in K
    pressure = dataset.data['P'] * 100 # in Pa
    RH = dataset.data['Relative_Humidity'] # wrt water, percent
    time =dataset.data['UTC_Start'] # seconds since midnight UTC
    rBC = dataset.data['BC_mass_90_550_nm'] # ng/m^3 STP
    rBC[rBC < 0] = np.nan
    
    ##--Constrain latitude to the Arctic region--##
    latitude[latitude < 66.5] = np.nan
    
    

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
        
    ##--Convert ptemp to np array--##
    potential_temp = np.array(potential_temp)
        
    ##--Constrain ptemp to range of other three campaigns--##
    potential_temp[potential_temp > 310] = np.nan
        
    
    ATom_conditions_dfs.append(pd.DataFrame({'temperature': temperature, 
                    'pressure': pressure, 'PTemp': potential_temp, 
                    'latitude': latitude, 'rBC': rBC, 'RH':RH}))
    
##--NETCARE--##
NETCARE_conditions_dfs = []
 
##--Loop through each flight, pulling and analyzing data--##
for flight in NETCARE_to_analyze:
    ##--Follow which flight is processing--##
    print(f"Processing {flight}...")

    ##--Pull meteorological data from AIMMS monitoring system--##
    aimms_files = find_files(NETCARE_directory, flight, "AIMMS_POLAR6")
    if aimms_files:
        aimms = icartt.Dataset(aimms_files[0])
    else:
        print(f"No AIMMS_POLAR6 file found for {flight}. Skipping...")
        continue  
 
    ##--Black carbon data from SP2--##
    SP2_files = find_files(NETCARE_directory, flight, "SP2_Polar6")
    if SP2_files: 
        SP2 = icartt.Dataset(SP2_files[0])
    else: 
        print(f"No SP2 file found for {flight}. Skipping...")
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
    RH = aimms.data['RH']*100 # % w.r.t. water
    
    
    ##--Constrain latitude to the Arctic region--##
    latitude[latitude < 66.5] = np.nan
    
    ##--Establish AIMMS start/stop times--##
    aimms_end = aimms_time.max()
    aimms_start = aimms_time.min()
    
    ##--Black carbon--##
    BC_count = SP2.data['BC_numb_concSTP'] # in STP

    ##--Handle black carbon data with different start/stop times than AIMMS--##
    BC_time = SP2.data['Time_UTC']

    ##--Trim CO data if it starts before AIMMS--##
    if BC_time.min() < aimms_start:
        mask_start = BC_time >= aimms_start
        BC_time = BC_time[mask_start]
        BC_count = BC_count[mask_start]
        
    ##--Append CO data with NaNs if it ends before AIMMS--##
    if BC_time.max() < aimms_end: 
        missing_times = np.arange(BC_time.max()+1, aimms_end +1)
        BC_time = np.concatenate([BC_time, missing_times])
        BC_count = np.concatenate([BC_count, [np.nan]*len(missing_times)])

    ##--Create a DataFrame for BC data and reindex to AIMMS time, setting non-overlapping times to nan--##
    BC_df = pd.DataFrame({'Time_UTC': BC_time, 'BC_count': BC_count})
    BC_aligned = BC_df.set_index('Time_UTC').reindex(aimms_time)
    BC_aligned['BC_count']= BC_aligned['BC_count'].where(BC_aligned.index.isin(aimms_time), np.nan)
    BC_count_aligned = BC_aligned['BC_count']

    rBC = BC_count_aligned.mask(BC_count_aligned < 0)
    
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
    
    ##--Place conditions in a separate df--##
    NETCARE_conditions_dfs.append(pd.DataFrame({'temperature': temperature, 
                'pressure': pressure, 'PTemp': potential_temp, 'RH': RH,
                'rBC':rBC, 'latitude': latitude}, index=aimms_time))
    
##--Store processed data here: --##
PAMARCMiP_conditions_dfs = []
 
##--Loop through each flight, pulling and analyzing data--##
for flight in PAMARCMiP_to_analyze:
    ##--Follow which flight is processing--##
    print(f"Processing {flight}...")
    
    ##--Pull file--##
    data = pd.read_csv(find_files(PAMARCMiP_directory, flight, ".csv")[0])
    
    #################
    ##--Pull data--##
    #################
    
    ##--Data--##
    altitude = data['Altitude'] # in m
    latitude = data['Latitude'] # in degrees
    temperature = data['Temp'] + 273.15 # in K
    pressure = data['Pressure'] # in pa
    time = data['Time'] # seconds since midnight
    rBC = data['SP2-Incand-Conc'].mask(data['SP2-Incand-Conc'] < 0) # ng/m^3?
    RH = data['RH'] # % w.r.t. water
    
    ##--The first datapoint in 'latitude' column is erraneous (47.12 N)--##
    ##--Constrain latitude to the Arctic region--##
    latitude = latitude.where(latitude >= 66.5, np.nan)

    
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
    
    ##--Append list of conditions--##
    ##--Place conditions in a separate df--##
    PAMARCMiP_conditions_dfs.append(pd.DataFrame({'temperature': temperature, 'RH':RH,
                            'pressure': pressure, 'PTemp': potential_temp, 
                            'rBC': rBC, 'latitude': latitude}))
    
FIREACE_conditions_dfs = []

for flight in FIREACE_to_analyze: 
    
    ##--Pull csv file containing all data--##
    files = find_files(FIREACE_directory, flight, "FIREACE")

    ##--The averaged data is always the second file--##
    if files:
        data = pd.read_csv(files[1])

    ##--Pull data variables from file--##
    time = data['Time'] # HHMMSS UTC time
    pressure = data['Pressure'] * 100 # in Pa
    temperature = data['Temperature'] + 273.15 # in K
    RH = data['RH'] # percent wrt water
    altitude = data['Altitude'] # in m (agl?)
    latitude = data['Latitude'] # degrees
    #longitude = data['Longitude'] # degrees
    
    ##--Constrain latitude to the Arctic region--##
    latitude = latitude.where(latitude >= 66.5, np.nan)
    
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

    ##--Make dummy array for rBC - no data for this campaign--##
    ##--Append conditions list--##
    FIREACE_conditions_dfs.append(pd.DataFrame({'temperature': temperature, 'RH':RH,
                                'pressure': pressure, 'rBC':np.full(len(time), np.nan),
                                'latitude': latitude, 'time': time, 'PTemp': potential_temp}))
    
############################
##--Create 2D histograms--##
############################

global_lat_edges = np.linspace(
    66.5,
    86.5,
    num_bins_lat + 1
)

global_ptemp_edges = np.linspace(
    235,
    310,
    num_bins_ptemp + 1
)

def compute_2d_median(df_list, value_col, lat_edges, ptemp_edges):

    if isinstance(df_list, pd.DataFrame):
        df_list = [df_list]

    all_lat = np.concatenate([df['latitude'].values for df in df_list])
    all_ptemp = np.concatenate([df['PTemp'].values for df in df_list])
    all_val = np.concatenate([df[value_col].values for df in df_list])

    mask = (~np.isnan(all_lat) &
            ~np.isnan(all_ptemp) &
            ~np.isnan(all_val))
    
    ##--Handle empty plots with no data--##
    if mask.sum() == 0:
     return np.full(
         (len(lat_edges)-1, len(ptemp_edges)-1),
         np.nan
     )

    stat, _, _, _ = binned_statistic_2d(
        all_lat[mask],
        all_ptemp[mask],
        all_val[mask],
        statistic="median",
        bins=[lat_edges, ptemp_edges]
    )

    return stat

campaigns = [
    ("2018", ATom_conditions_dfs),
    ("2015", NETCARE_conditions_dfs),
    #("2012", PAMARCMiP_conditions_dfs),
    ("1998", FIREACE_conditions_dfs),
]

################
##--Plotting--##
################

T_cmap = cm.cm.managua_r
RH_cmap = cm.cm.devon_r
rBC_cmap = cm.cm.grayC_r
 
fig, axes = plt.subplots(
    nrows=3,
    ncols=3,
    figsize=(15, 15),
    sharex=True,
    sharey=True,
    constrained_layout=True
)

for row, (name, df) in enumerate(campaigns):

    T_stat = compute_2d_median(df, "temperature",
                               global_lat_edges, global_ptemp_edges)

    RH_stat = compute_2d_median(df, "RH",
                                global_lat_edges, global_ptemp_edges)

    rBC_stat = compute_2d_median(df, "rBC",
                                 global_lat_edges, global_ptemp_edges)

    m1 = axes[row,0].pcolormesh(
        global_lat_edges,
        global_ptemp_edges,
        T_stat.T,
        shading="auto",
        cmap=T_cmap,
        vmin=200,
        vmax=300
    )
    
    m2 = axes[row,1].pcolormesh(
        global_lat_edges,
        global_ptemp_edges,
        RH_stat.T,
        shading="auto",
        cmap=RH_cmap,
        vmin=0, 
        vmax=100
    )
    
    m3 = axes[row,2].pcolormesh(
        global_lat_edges,
        global_ptemp_edges,
        rBC_stat.T,
        shading="auto",
        cmap=rBC_cmap,
        vmin=0,
        vmax=50
        #norm=mcolors.LogNorm(vmin=0.0000001, vmax=0.1)
    )

    axes[row,0].set_ylabel("\u0398 (K)", fontsize=22)

axes[0,0].set_title("Absolute Temperature", fontsize=26)
axes[0,1].set_title("Relative Humidity", fontsize=26)
axes[0,2].set_title("Refractive Black Carbon", fontsize=26)

##--Add sup labels for campaign year--##
axes[0, 0].text(-0.35, 0.5, "2018", verticalalignment='center', rotation=90, 
                fontsize=26, weight='bold', transform=axes[0,0].transAxes)
axes[1, 0].text(-0.35, 0.5, "2015", verticalalignment='center', fontsize=26, 
                rotation=90, weight='bold', transform=axes[1,0].transAxes)
axes[2, 0].text(-0.35, 0.5, "1998", verticalalignment='center', fontsize=26, 
                rotation=90, weight='bold', transform=axes[2,0].transAxes)

for ax in axes[-1,:]:
    ax.set_xlabel("Latitude (°)", fontsize=22)
    ax.tick_params(labelsize=18)
    ax.set_xlim(64.5, 87)
    ax.set_xticks([65, 70, 75, 80, 85])
    
for ax in axes[:,0]:
    ax.tick_params(axis='y', labelsize=18)
    ax.set_ylim(230, 315)
    
##--Add polar dome boundaries to NETCARE plots--##
for ax in axes[1,:]:
    ax.axhline(y=285, color='k', linestyle='--', linewidth=1)
    ax.axhline(y=299, color='k', linestyle='--', linewidth=1)
   
temp_cbar = fig.colorbar(m1, ax=axes[:,0], location='bottom', 
                         pad=0.01, shrink=0.95)
temp_cbar.set_label(label="Temperature (K)", fontsize=18, labelpad=22)
temp_cbar.ax.tick_params(labelsize=18, rotation=60)

RH_cbar = fig.colorbar(m2, ax=axes[:,1], ticks=[0, 20, 40, 60, 80, 100], 
                       location='bottom', pad=0.01, shrink=0.95)
RH_cbar.set_label(label="RH w.r.t. Water (%)", fontsize=18, labelpad=5)

##--Add + to last tick label--##
RH_cbar.set_ticklabels(["0", "20", "40", "60", "80", "100+"], 
                        fontsize=18, rotation=60)

rBC_cbar = fig.colorbar(m3, ax=axes[:,2], ticks=[0, 10, 20, 30, 40, 50], 
                       location='bottom', pad=0.01, shrink=0.95)
rBC_cbar.set_label(label="rBC concentration (ng/m$^{3}$)", fontsize=18, labelpad=15)

##--Add + to last tick label--##
rBC_cbar.set_ticklabels(["0", "10", "20", "30", "40", "50+"], 
                        fontsize=18, rotation=60)

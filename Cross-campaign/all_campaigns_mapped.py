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
from matplotlib.patches import Rectangle
import numpy as np

#########################
##--Open ICARTT Files--##
#########################
 
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
NETCARE_to_analyze = [
    "Flight1", 
    "Flight2", 
    "Flight3",
    "Flight4",
    "Flight5", 
    "Flight6", 
    "Flight7",
    "Flight8",
    "Flight9",
    "Flight10"
    ]

FIREACE_to_analyze = [ 
    "Flight3",
    "Flight7",
    "Flight8",
    "Flight9",
    "Flight10", 
    "Flight11", 
    "Flight12", 
    "Flight13", 
    "Flight14", 
    "Flight15", 
    "Flight16", 
    "Flight17", 
    "Flight18"
    ]

PAMARCMiP_to_analyze = [
    "Flight1", 
    "Flight2", 
    "Flight3",
    "Flight4",
    "Flight5", 
    "Flight6", 
    "Flight7",
    "Flight8",
    "Flight9"
    ]

ATom_to_analyze = [
    "Flight2", 
    "Flight10", 
    "Flight11", 
    "Flight12"
    ]

 
##--Store processed data here: --##
NETCARE_dfs = []
 
##--Loop through each flight, pulling and analyzing data--##
for flight in NETCARE_to_analyze:
    ##--Follow which flight is processing--##
    print(f"Processing NETCARE {flight}...")
    
    ##--Populate flight_dir established in above function--##
    NETCARE_dir = os.path.join(NETCARE, flight)
    
    ##--Pull meteorological data from AIMMS monitoring system--##
    aimms_files = find_files(NETCARE_dir, "AIMMS_POLAR6")
    if aimms_files:
        aimms = icartt.Dataset(aimms_files[0])
    else:
        print(f"No AIMMS_POLAR6 file found for {flight}. Skipping...")
        continue  # Skip to the next flight if AIMMS file is missing
 
    #################
    ##--Pull data--##
    #################
    
    ##--AIMMS Data--##
    NETCARE_altitude = aimms.data['Alt'] # in m
    NETCARE_latitude = aimms.data['Lat'] # in degrees
    NETCARE_longitude = aimms.data['Lon'] # in degrees

    NETCARE_df = pd.DataFrame({'Lat': NETCARE_latitude,'Lon': NETCARE_longitude, 'Alt': NETCARE_altitude})
    NETCARE_dfs.append(NETCARE_df[['Lat', 'Lon', 'Alt']])
    
##--Repeate for FIRE-ACE 1998--##
FIREACE_dfs = []

for flight in FIREACE_to_analyze:  
    print(f"Processing FIREACE {flight}...")    
    
    FIREACE_dir = os.path.join(FIREACE, flight)
    
    data_files = find_files(FIREACE_dir, "FIREACE")
    if data_files:
        data = pd.read_csv(data_files[0])
    else:
        print(f"No file found for {flight}. Skipping...")
        continue  
        
    #################
    ##--Pull data--##
    #################
    
    ##--AIMMS Data--##
    FIREACE_altitude = data['Altitude'] # in m
    FIREACE_latitude = data['Latitude'] # in degrees
    FIREACE_longitude = data['Longitude'] # in degrees

    FIREACE_df = pd.DataFrame({'Lat': FIREACE_latitude,'Lon': FIREACE_longitude, 'Alt': FIREACE_altitude})
    
    FIREACE_df['Lon'] = FIREACE_df['Lon'] * -1
    
    FIREACE_dfs.append(FIREACE_df[['Lat', 'Lon', 'Alt']])
    
##--Repeate for PAMARCMiP 2012--##
PAMARCMiP_dfs = []

for flight in PAMARCMiP_to_analyze:  
    print(f"Processing PAMARCMiP {flight}...")    
    
    PAMARCMiP_dir = os.path.join(PAMARCMiP, flight)
    
    p_data_files = find_files(PAMARCMiP_dir, "PA")
    if p_data_files:
        p_data = pd.read_csv(p_data_files[0])
    else:
        print(f"No file found for {flight}. Skipping...")
        continue  
        
    #################
    ##--Pull data--##
    #################
    
    ##--AIMMS Data--##
    PAMARCMiP_altitude = p_data['Altitude'] # in m
    PAMARCMiP_latitude = p_data['Latitude'] # in degrees
    PAMARCMiP_longitude = p_data['Longitude'] # in degrees

    PAMARCMiP_df = pd.DataFrame({'Lat': PAMARCMiP_latitude,'Lon': PAMARCMiP_longitude, 'Alt': PAMARCMiP_altitude})
    
    PAMARCMiP_df['Lon'] = PAMARCMiP_df['Lon']
    
    PAMARCMiP_dfs.append(PAMARCMiP_df[['Lat', 'Lon', 'Alt']])

##--Repeate for ATom 2018--##
ATom_dfs = []

for flight in ATom_to_analyze:  
    print(f"Processing PAMARCMiP {flight}...")    
    
    ATom_dir = os.path.join(ATom, flight)
    
    ATom_files = find_files(ATom_dir, "MER")
    if ATom_files:
        data = icartt.Dataset(ATom_files[0])
    else:
        print(f"No file found for {flight}. Skipping...")
        continue  
        
    #################
    ##--Pull data--##
    #################
    
    ##--AIMMS Data--##
    ATom_altitude = data.data['G_ALT'] # in m
    ATom_latitude = data.data['LAT_AMSSD'] # in degrees
    ATom_longitude = data.data['LON_AMSSD'] # in degrees
    
    ##--Constrain latitude to the Arctic region--##
    ATom_latitude[ATom_latitude < 66.5] = np.nan

    ATom_df = pd.DataFrame({'Lat': ATom_latitude,'Lon': ATom_longitude, 'Alt': ATom_altitude})
    
    ATom_df['Lon'] = ATom_df['Lon']
    
    ATom_dfs.append(ATom_df[['Lat', 'Lon', 'Alt']])
        
 
################
##--Plotting--##
################

##--Create the map with a North Polar Stereo projection--##
fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=-90)})

##--Add land and ocean features--##
land = cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='black', facecolor='lightgray')
ocean = cfeature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='none', facecolor='lightblue')
ax.add_feature(ocean)
ax.add_feature(land)
ax.coastlines()
ax.set_extent([-180, 180, 60, 90], crs=ccrs.PlateCarree())
'''
##--Clip the map to desired range without changing extent--##
clip_rect = Rectangle(
    (-180, 65),          # lower-left corner
    360,                 # width (180 - -180)
    10,                  # height (75 - 65)
    transform=ccrs.PlateCarree()
)

# Apply the clip path to all plotted collections
for coll in ax.collections:
    coll.set_clip_path(clip_rect)
'''
##--ATom--##
for df in ATom_dfs:
    ax.scatter(df['Lon'], df['Lat'], color='purple',
               transform=ccrs.PlateCarree(), s=20, zorder=3)

ax.scatter([], [], color='purple', label='ATom 2018', s=16)
   
##--Plot all NETCARE flights--##
for df in NETCARE_dfs:
    ax.scatter(df['Lon'], df['Lat'], color='teal',
               transform=ccrs.PlateCarree(), s=20, zorder=3)
    
##--Empty plot used to set up legend for NETCARE--##
ax.scatter([], [], color='teal', label='NETCARE 2015', s=16)

##--PAMARCMiP--##
for df in PAMARCMiP_dfs:
    ax.scatter(df['Lon'], df['Lat'], color='olive',
               transform=ccrs.PlateCarree(), s=20, zorder=3)

ax.scatter([], [], color='olive', label='PAMARCMiP 2012', s=16)

##--FIREACE--##
for df in FIREACE_dfs:
    ax.scatter(df['Lon'], df['Lat'], color='goldenrod',
               transform=ccrs.PlateCarree(), s=20, zorder=3)

ax.scatter([], [], color='goldenrod', label='FIRE-ACE 1998', s=16)

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
    ax.scatter(lon, lat, color='red', marker='*', s=250, edgecolor='black', transform=ccrs.PlateCarree(), zorder=4)


'''
##--Create insets in matplotlip using add_axes--##

##--Adjust placement: first param is left-right set, second up-down--##
##--Adjust size: third param adjusts x-axis relative scale, fourth y-axis relative scale--##
ax_inset1 = fig.add_axes([0.60, 0.11, 0.25, 0.25], projection=ccrs.NorthPolarStereo(central_longitude=-90))

##--Create an inset map for flights 2-7 (Near Eureka and Alert)--##
ax_inset1.set_extent([-100, -68, 79, 84], crs=ccrs.PlateCarree())  # Adjust region
ax_inset1.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax_inset1.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray')
#ax_inset1.coastlines()

##--Plot flight tracks on first inset--##
for i, (flight, df) in enumerate(zip(flights_to_analyze, Flight_dfs)):
    ax_inset1.scatter(df['Lon'], df['Lat'], color=cmap(norm(i)), label=flight,
               transform=ccrs.PlateCarree(), s=5, zorder=3)

for name, (lon, lat) in locations.items():
    ax_inset1.scatter(lon, lat, color='red', marker='*', s=100, edgecolor='black', transform=ccrs.PlateCarree(), zorder=4)
 
ax_inset2 = fig.add_axes([0.12, 0.25, 0.20, 0.20], projection=ccrs.NorthPolarStereo(central_longitude=-90))
    
##--Create an inset map for flights 8-10 (Near Inuvik)--##
ax_inset2.set_extent([-134, -129, 65, 72], crs=ccrs.PlateCarree())  # Adjust region
ax_inset2.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax_inset2.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray')
ax_inset2.coastlines()

##--Plot flight tracks on second inset--##
for i, (flight, df) in enumerate(zip(flights_to_analyze, Flight_dfs)):
    ax_inset2.scatter(df['Lon'], df['Lat'], color=cmap(norm(i)), label=flight,
               transform=ccrs.PlateCarree(), s=5, zorder=3)

for name, (lon, lat) in locations.items():
    ax_inset2.scatter(lon, lat, color='red', marker='*', s=100, edgecolor='black', transform=ccrs.PlateCarree(), zorder=4)

ax_inset3 = fig.add_axes([0.66, 0.43, 0.20, 0.20], projection=ccrs.NorthPolarStereo(central_longitude=-90))

##--Create an inset map for flight 1 (Near Ny Alesund)--##
ax_inset3.set_extent([3, 18, 77, 81], crs=ccrs.PlateCarree())  # Adjust region
ax_inset3.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax_inset3.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray')
ax_inset3.coastlines()

##--Plot flight tracks on third inset--##
for i, (flight, df) in enumerate(zip(flights_to_analyze, Flight_dfs)):
    ax_inset3.scatter(df['Lon'], df['Lat'], color=cmap(norm(i)), label=flight,
               transform=ccrs.PlateCarree(), s=5, zorder=3)

for name, (lon, lat) in locations.items():
    ax_inset3.scatter(lon, lat, color='red', marker='*', s=100, edgecolor='black', transform=ccrs.PlateCarree(), zorder=4)
'''
##--Add legend for flight numbers--##
legend = ax.legend(loc='upper right', fontsize=16, framealpha=1, markerscale=3, ncol=1)

##--Add gridlines--##
gl = ax.gridlines(draw_labels=True)
gl.top_labels = True
gl.bottom_labels = True
gl.xlabel_style = {'size': 16}   # longitude labels
gl.ylabel_style = {'size': 16}   # latitude labels

##--Base output path in directory--##
#output_path = r"C:\Users\repooley\REP_PhD\NETCARE2015\Arctic_NPF\data\processed\MappedData\Mapped_Flights.png"

#plt.savefig(output_path, dpi=300, bbox_inches='tight') 

plt.show()

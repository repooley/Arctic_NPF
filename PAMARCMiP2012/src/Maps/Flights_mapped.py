# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 10:33:24 2026

@author: repooley
"""

import os
import glob
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

#########################
##--Open ICARTT Files--##
#########################

PAMARCMiP = r"C:\Users\repooley\REP_PhD\Arctic_NPF\PAMARCMiP2012\data\raw"
 
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
    
 
################
##--Plotting--##
################

##--Create the map with a North Polar Stereo projection--##
fig, ax = plt.subplots(figsize=(8, 12), subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=-90)})

##--Add land and ocean features--##
land = cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='black', facecolor='lightgray')
ocean = cfeature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='none', facecolor='lightblue')
ax.add_feature(ocean)
ax.add_feature(land)
ax.coastlines()
ax.set_extent([-180, 120, 70, 90], crs=ccrs.PlateCarree())


##--PAMARCMiP--##
for df in PAMARCMiP_dfs:
    ax.scatter(df['Lon'], df['Lat'], color='purple',
               transform=ccrs.PlateCarree(), s=20, zorder=3)

ax.scatter([], [], color='purple', label='PAMARCMiP 2012', s=20)

    
##--Add locations with star markers and labels--##
locations = {
    "Alert, NU": (-62.34, 82.50),
    "Eureka, NU": (-85.93, 79.98),
    "Resolute, NU": (-94.8292, 74.6885)
}

for name, (lon, lat) in locations.items():
    ax.scatter(lon, lat, color='red', marker='*', s=250, edgecolor='black', transform=ccrs.PlateCarree(), zorder=4)

##--Add legend for flight numbers--##
legend = ax.legend(loc='upper right', fontsize=20, framealpha=1, markerscale=3, ncol=1)

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

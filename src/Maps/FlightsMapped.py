# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 10:15:04 2025

@author: repooley
"""
import icartt
import os
import glob
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

#########################
##--Open ICARTT Files--##
#########################
 
##--Set the base directory to project folder--##
directory = r"C:\Users\repooley\REP_PhD\NETCARE2015\data\raw"
 
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
 
##--Choose which flights (1-10) to analyze here! Comment out unwanted--##
flights_to_analyze = [
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
 
##--Store processed data here: --##
Flight_dfs = []
 
##--Loop through each flight, pulling and analyzing data--##
for flight in flights_to_analyze:
    ##--Follow which flight is processing--##
    print(f"Processing {flight}...")
    
    ##--Populate flight_dir established in above function--##
    flight_dir = os.path.join(directory, flight)
    
    ##--Pull meteorological data from AIMMS monitoring system--##
    aimms_files = find_files(flight_dir, "AIMMS_POLAR6")
    if aimms_files:
        aimms = icartt.Dataset(aimms_files[0])
    else:
        print(f"No AIMMS_POLAR6 file found for {flight}. Skipping...")
        continue  # Skip to the next flight if AIMMS file is missing
 
    #################
    ##--Pull data--##
    #################
    
    ##--AIMMS Data--##
    altitude = aimms.data['Alt'] # in m
    latitude = aimms.data['Lat'] # in degrees
    longitude = aimms.data['Lon'] # in degrees

    Flight_df = pd.DataFrame({'Lat': latitude,'Lon': longitude, 'Alt': altitude})
    Flight_dfs.append(Flight_df[['Lat', 'Lon', 'Alt']])
        
################
##--Plotting--##
################

##--Create the map with a North Polar Stereo projection--##
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=-90)})

##--Add land and ocean features--##
land = cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='black', facecolor='lightgray')
ocean = cfeature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='none', facecolor='lightblue')
ax.add_feature(ocean)
ax.add_feature(land)
ax.coastlines()
ax.set_extent([-180, 180, 55, 90], crs=ccrs.PlateCarree())

##--Generate distinct colors for flights--##
cmap = plt.cm.get_cmap('viridis', len(Flight_dfs))  
norm = mcolors.Normalize(vmin=0, vmax=len(Flight_dfs) - 1)

##--Plot all flight tracks--##
for i, (flight, df) in enumerate(zip(flights_to_analyze, Flight_dfs)):
    ##--Include space in label between "Flight" and number--##
    ax.scatter(df['Lon'], df['Lat'], color=cmap(norm(i)), label=f"Flight {flight.replace('Flight', '')}",
               transform=ccrs.PlateCarree(), s=20, zorder=3)
    
##--Add locations with star markers and labels--##
locations = {
    "Alert, NU": (-62.34, 82.50),
    "Eureka, NU": (-85.93, 79.98),
    "Inuvik, NWT": (-133.72, 68.36),
    "Ny Alesund, NO": (11.99, 78.93)
}

for name, (lon, lat) in locations.items():
    ax.scatter(lon, lat, color='red', marker='*', s=100, edgecolor='black', transform=ccrs.PlateCarree(), zorder=4)

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

##--Add legend for flight numbers--##
legend = ax.legend(loc='upper center', fontsize=12, framealpha=1, markerscale=2, ncol=3)

##--Add gridlines--##
gl = ax.gridlines(draw_labels=True)
gl.top_labels = True
gl.bottom_labels = True

##--Base output path in directory--##
output_path = r"C:\Users\repooley\REP_PhD\NETCARE2015\data\processed\MappedData\Mapped_Flights.png"

plt.savefig(output_path, dpi=300, bbox_inches='tight') 

plt.show()

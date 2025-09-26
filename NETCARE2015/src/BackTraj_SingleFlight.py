# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 13:01:02 2025

@author: repooley
"""
##--This script was written referencing @eli's work--##

import os
import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
from matplotlib.dates import DateFormatter, date2num
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import alphashape
from scipy.spatial import ConvexHull

###################
##--User inputs--##
###################

##--Set the base directory to project folder--##
hysplit = r"C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data\raw\HYSPLIT\data\trajectories"
 
##--Select flight (Flight1 thru Flight10)--##
flight = "Flight5" 

##--Filter to above the polar dome?--##
above_dome = True 

##--Base output path for figures in directory--##
output_path = r"C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data\processed\HYSPLIT"

Netcare = pd.read_csv(r"C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data\raw\Netcare.csv")

##################
##--Pull Files--##
##################

##--Create directory based on selected flight--##
def find_files(directory, flight):
    flight_dir = os.path.join(hysplit, flight)
    return flight_dir

flight_directory = find_files(hysplit, flight)

##--Get timestamps where trajectories were initialized--##
##--Trajectories were initialized every 10 minutes from the Netcare file--##
single_flight = Netcare[Netcare['Flight_num'] == flight]

start_utc = int(single_flight['Time_start'].min())
end_utc = int(single_flight['Time_start'].max())
UTCs = list(range(start_utc, end_utc +1, 600))

##--Subset Netcare to times in UTCs--##
netcare_subset = single_flight[single_flight['Time_start'].isin(UTCs)]

###################
##--Set up plot--##
###################

##--Flight-by-flight parameters--##
if flight == "Flight2":
    map_extent = [-120, 10, 35, 90]
    height_ratios = [2, 1]
    hspace = 0.1
    htitle = 0.92
elif flight == ["Flight3", "Flight5"]:
    map_extent = [-120, 10, 70, 90]
    height_ratios = [2, 1]
    hspace = 0
    htitle=0.91
elif flight == "Flight4":
    map_extent = [-120, 15, 60, 90]
    height_ratios = [2, 1]
    hspace = 0.1
    htitle=0.91
elif flight == "Flight6":
    map_extent = [-120, 10, 65, 90]
    height_ratios = [2, 1]
    hspace = 0.1
    htitle=0.91
elif flight == "Flight7":
    map_extent = [-120, 32, 45, 90]
    height_ratios = [2.5, 1]
    hspace = 0.0
    htitle=0.85
elif flight == "Flight8":
    map_extent = [-180, 120, 28, 90]
    height_ratios = [2, 1]
    hspace = 0.0
    htitle=0.85
elif flight == "Flight9":
    map_extent = [-180, 60, 28, 90]
    height_ratios = [2, 1]
    hspace = -0.1
    htitle=0.88
elif flight == "Flight10":
    map_extent = [-180, 60, 28, 90]
    height_ratios = [2, 1]
    hspace = -0.1
    htitle=0.88
else: 
    map_extent = [-120, 10, 70, 90]
    height_ratios = [2, 1]
    hspace = 0.1
    htitle=0.91
    
##--Separate the axes from the figure object to apply different projections--##
fig = plt.figure(figsize=(12, 10))

##--Use gridspec to access figure layout--##
gs = gridspec.GridSpec(2, 2, height_ratios=height_ratios, hspace=hspace)

##--Create a map with polar stereographic projection in first subplot--##
##--Top row: NPF is significant--##
ax_map_sig = fig.add_subplot(gs[0, 0], projection=ccrs.NorthPolarStereo())
ax_time_sig = fig.add_subplot(gs[1, 0])

##--Bottom row: non-significant NPF--##
ax_map_nonsig = fig.add_subplot(gs[0, 1], projection=ccrs.NorthPolarStereo())
ax_time_nonsig = fig.add_subplot(gs[1, 1])

##--Add map features--##
for ax_map in [ax_map_sig, ax_map_nonsig]:
    ax_map.set_extent(map_extent, crs=ccrs.PlateCarree())
    ax_map.add_feature(cfeature.LAND, fc='darkseagreen', ec='k', lw=0.2, zorder=2)
    ax_map.add_feature(cfeature.OCEAN, fc='lightblue', ec='k', lw=0.2, zorder=1)

##--Set labels--##
ax_map_sig.set_title("$N_{2.5-10}$ Event", fontsize=14)
ax_map_nonsig.set_title("No Significant $N_{2.5-10}$", fontsize=14)
ax_time_sig.set_ylabel("Meters Above Ground Level", fontsize=12)
#ax_time_nonsig.set_ylabel("Meters Above Ground Level", fontsize=12)

fig.suptitle(f"NETCARE {flight.replace('Flight', 'Flight ')} HYSPLIT Back Trajectories", 
             fontsize=16, y=htitle)

##--Set axes limits--##
ax_time_sig.set_ylim(-250, 8000)
ax_time_nonsig.set_ylim(-250, 8000)


lats_sig, lons_sig = [], []
lats_nonsig, lons_nonsig = [], []

alt_sig, time_sig = [], []
alt_nonsig, time_nonsig = [], []

##########################
##--Loop through files--##
##########################

for file, row in zip(sorted(os.listdir(flight_directory)), netcare_subset.itertuples(index=False)):
    
    ##--Condition: initialized traj must be above the marginal polar dome--##
    ##--Marginal dome boundary is 285 K--##
    if above_dome == True: 
        if row.ptemp > 285:
    
            ##--Determine which axis to use (NPF vs non-NPF)--##
            is_significant = pd.notna(row.nuc_significant)
            ax_map = ax_map_sig if is_significant else ax_map_nonsig
            ax_time = ax_time_sig if is_significant else ax_time_nonsig
                
            ##--\s denotes any whitespace character, + indicates one or more spaces--##
            df = pd.read_csv(os.path.join(hysplit, flight, file), sep=r'\s+')
            
            ##--Rename DATE to DAY--##
            df = df.rename(columns={'DATE': 'DAY'})
            
            ##--Change year to four digits, .apply() takes a function as an argument--##
            ##--A lambda function is local only--##
            df['YEAR'] = df['YEAR'].apply(lambda y: y + 2000)
            
            ##--Format for year, month, day, hour--##
            df['DateTime'] = pd.to_datetime({'year': df['YEAR'], 'month': df['MONTH'],
                    'day': df['DAY'], 'hour': df['HOUR']})
          
            ##--Group by TRAJ to place each trajectory in time order--##
            for traj_num, group in df.groupby('TRAJ'):
                group = group.sort_values('DateTime')
                
                altitudes = group['ALTITUDE'].values
                
                ##--Last time in trajectory = initialization (measurement) time--##
                t0 = group['DateTime'].iloc[-1]
                
                ##--Compute relative time in days (backward from initialization)--##
                ##--Divide by length of one day: 86400 seconds--##
                time_rel = (group['DateTime'] - t0).dt.total_seconds() / 86400.0
        
                ##--Cut off trajectory within 1m of surface, HYSPLIT is iffy here--##
                if any(altitudes < 1):
                    index_end = np.min(np.where(altitudes < 1))
                else:
                    index_end = len(group) 
                    
               ##--Plot original (unperturbed) trajectory--##
                if traj_num == 1:
                    color = 'k'
                    linewidth = 0.75
                    alpha = 1
                    zorder = 5
                else:
                    color = 'none'
                
                ax_map.plot(group['LONG'].iloc[:index_end],
                    group['LAT'].iloc[:index_end],
                    transform=ccrs.PlateCarree(),
                    c=color, lw=linewidth, alpha=alpha, zorder=zorder)
                
                ax_time.plot(time_rel.iloc[:index_end],
                    group['ALTITUDE'].iloc[:index_end],
                    c=color, lw=linewidth, alpha=alpha, zorder=zorder)
                
                if is_significant:
                    lats_sig.extend(group['LAT'].values)
                    lons_sig.extend(group['LONG'].values)
                    alt_sig.extend(group['ALTITUDE'].values)
                    time_sig.extend(time_rel.values) 
                else:
                    lats_nonsig.extend(group['LAT'].values)
                    lons_nonsig.extend(group['LONG'].values)
                    alt_nonsig.extend(group['ALTITUDE'].values)
                    time_nonsig.extend(time_rel.values) 
    else: 
        
        ##--Determine which axis to use (NPF vs non-NPF)--##
        is_significant = pd.notna(row.nuc_significant)
        ax_map = ax_map_sig if is_significant else ax_map_nonsig
        ax_time = ax_time_sig if is_significant else ax_time_nonsig
            
        ##--\s denotes any whitespace character, + indicates one or more spaces--##
        df = pd.read_csv(os.path.join(hysplit, flight, file), sep=r'\s+')
        
        ##--Rename DATE to DAY--##
        df = df.rename(columns={'DATE': 'DAY'})
        
        ##--Change year to four digits, .apply() takes a function as an argument--##
        ##--A lambda function is local only--##
        df['YEAR'] = df['YEAR'].apply(lambda y: y + 2000)
        
        ##--Format for year, month, day, hour--##
        df['DateTime'] = pd.to_datetime({'year': df['YEAR'], 'month': df['MONTH'],
                'day': df['DAY'], 'hour': df['HOUR']})
      
        ##--Group by TRAJ to place each trajectory in time order--##
        for traj_num, group in df.groupby('TRAJ'):
            group = group.sort_values('DateTime')
            
            altitudes = group['ALTITUDE'].values
            
            ##--Last time in trajectory = initialization (measurement) time--##
            t0 = group['DateTime'].iloc[-1]
            
            ##--Compute relative time in days (backward from initialization)--##
            ##--Divide by length of one day: 86400 seconds--##
            time_rel = (group['DateTime'] - t0).dt.total_seconds() / 86400.0
    
            ##--Cut off trajectory within 1m of surface, HYSPLIT is iffy here--##
            if any(altitudes < 1):
                index_end = np.min(np.where(altitudes < 1))
            else:
                index_end = len(group) 
                
           ##--Plot original (unperturbed) trajectory--##
            if traj_num == 1:
                color = 'k'
                linewidth = 0.75
                alpha = 1
                zorder = 5
            else:
                color = 'none'
            
            ax_map.plot(group['LONG'].iloc[:index_end],
                group['LAT'].iloc[:index_end],
                transform=ccrs.PlateCarree(),
                c=color, lw=linewidth, alpha=alpha, zorder=zorder)
            
            ax_time.plot(time_rel.iloc[:index_end],
                group['ALTITUDE'].iloc[:index_end],
                c=color, lw=linewidth, alpha=alpha, zorder=zorder)
            
            if is_significant:
                lats_sig.extend(group['LAT'].values)
                lons_sig.extend(group['LONG'].values)
                alt_sig.extend(group['ALTITUDE'].values)
                time_sig.extend(time_rel.values) 
            else:
                lats_nonsig.extend(group['LAT'].values)
                lons_nonsig.extend(group['LONG'].values)
                alt_nonsig.extend(group['ALTITUDE'].values)
                time_nonsig.extend(time_rel.values)    

##--Set up function to grey out empty plots--##
def grey_plots(ax):
    
    ##--Add a grey rectangle covering the whole axes--##
    ax.add_patch(mpatches.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                 facecolor='lightgrey', alpha=0.6, zorder=7))
            
##########################################
##--Map: lat/lon binned traj frequency--##
##########################################

##--INPUTS--##

##--The alpha value controls the hull fit--##
alpha = 0.2

##--Set min/max latitudes for binning in degrees--##
lat_min = 30    
lat_max = 90  

##--Set number of lon/lat bins--##
num_theta = 150  # longitude bins
num_r = 250       # latitude bins      

##--Longitude edges (uniform degrees)--##
lon_edges = np.linspace(-180, 180, num_theta + 1)

##--Latitude edges uniform in sin(lat) for equal area--##
sin_edges = np.linspace(np.sin(np.deg2rad(lat_min)),
                        np.sin(np.deg2rad(lat_max)),
                        num_r + 1)
lat_edges = np.rad2deg(np.arcsin(sin_edges))


##--Histograms in lon/lat space--##
H_sig, _, _ = np.histogram2d(lons_sig, lats_sig, bins=(lon_edges, lat_edges))
H_nonsig, _, _ = np.histogram2d(lons_nonsig, lats_nonsig, bins=(lon_edges, lat_edges))

##--Compute bin areas on a unit sphere--##
lon_rads = np.deg2rad(lon_edges)
lat_rads = np.deg2rad(lat_edges)

dlam = np.diff(lon_rads)                 # Δλ for each longitude bin
dsinphi = np.diff(np.sin(lat_rads))      # Δ(sin φ) for each latitude bin

areas = np.outer(dlam, dsinphi)          # (num_theta, num_r) bin areas

##--Normalize histograms to counts per unit area--##
H_sig_density = H_sig / areas
H_nonsig_density = H_nonsig / areas

##--Project bin edges into stereographic (x, y)--##
def stereographic_proj(lon_deg, lat_deg, lon0=0):
    """North-pole stereographic projection (R=1)."""
    lon = np.deg2rad(lon_deg)
    lat = np.deg2rad(lat_deg)
    lon0 = np.deg2rad(lon0)

    k = 2 / (1 + np.sin(lat) * np.sin(np.pi/2) +
             np.cos(lat) * np.cos(np.pi/2) * np.cos(lon - lon0))
    x = k * np.cos(lat) * np.sin(lon - lon0)
    y = k * (np.cos(np.pi/2) * np.sin(lat) -
             np.sin(np.pi/2) * np.cos(lat) * np.cos(lon - lon0))
    return x, y


lon_grid, lat_grid = np.meshgrid(lon_edges, lat_edges, indexing="ij")
x_edges, y_edges = stereographic_proj(lon_grid, lat_grid)

##--Check if H_sig is empty--##
if H_sig.size > 0 and H_sig.sum() > 0:
    ##--Compute percent frequency for each bin--##
    H_sig_percent = 100 * H_sig / H_sig.sum()
    
    ##--Mask bins with zero frequency--##
    H_sig_masked = ma.masked_where(H_sig_percent == 0, H_sig_percent)
    
    ##--Plot lat/lon density--##
    bin_colors = ax_map_sig.pcolormesh(lon_grid, lat_grid, H_sig_masked, cmap='plasma', 
        alpha=0.25, edgecolors='none', transform=ccrs.PlateCarree(), zorder=4)
    
    ##--Flatten the histograms--##
    flat_sig = H_sig_percent.flatten()
    
    ##--Sort bins by frequency--##
    ##--Sort by percent, not xy location--##
    sorted_sig_idx = np.argsort(flat_sig)[::-1]  # descending order
    ##--Cumulate all bins--##
    cum_sig = np.cumsum(flat_sig[sorted_sig_idx])
    
    ##--Establish lower cutoff for bins with lower 10% of data based on index--##
    cutoff_sig_idx = np.where(cum_sig <= 90)[0] 
    keep_sig_idx = sorted_sig_idx[cutoff_sig_idx]
    
    ##--Create mask with selected bins above cutoff--##
    mask_sig_top90 = np.zeros_like(flat_sig, dtype=bool) # establish zeroes
    mask_sig_top90[keep_sig_idx] = True # index the mask
    mask_sig_top90 = mask_sig_top90.reshape(H_sig_percent.shape) # shape the mask
    
    selected_sig_boxes = []
    for i in range(mask_sig_top90.shape[0]):
        for j in range(mask_sig_top90.shape[1]):
            if mask_sig_top90[i, j]:
                lon_center = 0.25 * (lon_grid[i, j] + lon_grid[i+1, j] +
                                     lon_grid[i+1, j+1] + lon_grid[i, j+1])
                lat_center = 0.25 * (lat_grid[i, j] + lat_grid[i+1, j] +
                                     lat_grid[i+1, j+1] + lat_grid[i, j+1])
                selected_sig_boxes.append((lon_center, lat_center))
    
    ##--Select bins to draw a hull polygon around--##
    points_sig_bins = np.unique(selected_sig_boxes, axis=0)   
    
   ##--Plot hull polygon--##
    if len(points_sig_bins) >= 3:  # alphashape also needs at least 3 points
        hull_sig_bins = alphashape.alphashape(points_sig_bins, alpha)
   
        if hull_sig_bins.geom_type == "Polygon":
            x, y = hull_sig_bins.exterior.xy
            ax_map_sig.fill(x, y, facecolor='None', edgecolor='orangered', ls='--', lw=2,
                           transform=ccrs.PlateCarree(), zorder=5)
   
        elif hull_sig_bins.geom_type == "MultiPolygon":
            for poly in hull_sig_bins.geoms:
                if not poly.is_empty:
                    x, y = poly.exterior.xy
                    ax_map_sig.fill(x, y, facecolor='None', edgecolor='orangered', ls='--', lw=2,
                                   transform=ccrs.PlateCarree(), zorder=5)
   
        else:
            print("Alpha shape is not a polygon:", hull_sig_bins.geom_type)
   
    else:
        H_sig_percent = np.zeros_like(H_sig)   # keep same shape
        H_sig_masked  = ma.masked_all(H_sig.shape)
        points_sig_bins = np.empty((0, 2))     # safe empty 2D


##--H_nonsig should always have values--##

H_nonsig_percent = 100 * H_nonsig / H_nonsig.sum()   
 
H_nonsig_masked = ma.masked_where(H_nonsig_percent == 0, H_nonsig_percent)

bin_colors = ax_map_nonsig.pcolormesh(lon_grid, lat_grid, H_nonsig_masked, cmap='plasma', 
    alpha=0.25, edgecolors='none', transform=ccrs.PlateCarree(), zorder=4)

flat_nonsig = H_nonsig_percent.flatten()

sorted_nonsig_idx = np.argsort(flat_nonsig)[::-1]
cum_nonsig = np.cumsum(flat_nonsig[sorted_nonsig_idx])

cutoff_nonsig_idx = np.where(cum_nonsig <= 90)[0]
keep_nonsig_idx = sorted_nonsig_idx[cutoff_nonsig_idx]

mask_nonsig_top90 = np.zeros_like(flat_nonsig, dtype=bool)
mask_nonsig_top90[keep_nonsig_idx] = True
mask_nonsig_top90 = mask_nonsig_top90.reshape(H_nonsig_percent.shape)

selected_nonsig_boxes = []
for i in range(mask_nonsig_top90.shape[0]):  
    for j in range(mask_nonsig_top90.shape[1]):  
        if mask_nonsig_top90[i, j]:
            lon_corners = [lon_grid[i, j], lon_grid[i+1, j], lon_grid[i+1, j+1], lon_grid[i, j+1]]
            lat_corners = [lat_grid[i, j], lat_grid[i+1, j], lat_grid[i+1, j+1], lat_grid[i, j+1]]
            for lon, lat in zip(lon_corners, lat_corners):
                selected_nonsig_boxes.append((lon, lat))

##--Select points to draw a hull polygon around--##
points_nonsig_bins = np.unique(selected_nonsig_boxes, axis=0)

##--Generate the alpha hull--##
hull_nonsig_bins = alphashape.alphashape(points_nonsig_bins, alpha)

##--From Google AI response--##
if hull_nonsig_bins.geom_type == "Polygon":
    x, y = hull_nonsig_bins.exterior.xy
    ax_map_nonsig.fill(x, y, facecolor='None', edgecolor='orangered', ls='--',
                       linewidth=2.5, transform=ccrs.PlateCarree(), zorder=7)

elif hull_nonsig_bins.geom_type == "MultiPolygon":
    for poly in hull_nonsig_bins.geoms:
        if not poly.is_empty:
            x, y = poly.exterior.xy
            ax_map_nonsig.fill(x, y, facecolor='None', edgecolor='orangered', ls='--', 
                       linewidth=3, transform=ccrs.PlateCarree(), zorder=7)
else:
    print("Alpha shape is not a polygon:", hull_nonsig_bins.geom_type)

   
##--Add one colorbar--##
cbar = plt.colorbar(bin_colors, ax=[ax_map_sig, ax_map_nonsig], orientation='vertical', shrink=0.5)
cbar.ax.tick_params(labelsize=14)
cbar.set_label('% Trajectory Frequency', size=12)


################
##--Altitude--##
################

##--These bin numbers apply to ALL curtain plots--##
num_time_bins = 12
num_alt_bins = 10

##--Convert altitude lists to arrays--##
alt_sig_arr = np.array(alt_sig)
alt_nonsig_arr = np.array(alt_nonsig)

##--Determine overall min/max relative times for binning--##
# relative time: 0 = measurement time, negative = days before
if len(time_sig) > 0:
    all_time_rel = np.concatenate([time_sig, time_nonsig])
else:
    all_time_rel = np.array(time_nonsig)

time_min = all_time_rel.min()  # earliest day
time_max = 0                   # measurement time = 0 days

##--Create bin edges in the time dimension--##
time_bins_rel = np.linspace(time_min, time_max, num_time_bins + 1)

##--Create altitude bin edges--##
min_alt = min(alt_sig_arr.min() if len(alt_sig_arr) > 0 else alt_nonsig_arr.min(),
              alt_nonsig_arr.min())
max_alt = max(alt_sig_arr.max() if len(alt_sig_arr) > 0 else alt_nonsig_arr.max(),
              alt_nonsig_arr.max())
alt_bins = np.linspace(min_alt, max_alt, num_alt_bins + 1)

##--Compute 2d histograms--##
if len(time_sig) > 0 and len(alt_sig_arr) > 0:
    H_sig, _, _ = np.histogram2d(time_sig, alt_sig_arr, bins=(time_bins_rel, alt_bins))
    H_sig_percent = 100 * H_sig / H_sig.sum()
    H_sig_masked = ma.masked_where(H_sig_percent == 0, H_sig_percent)
    
    mesh_sig = ax_time_sig.pcolormesh(time_bins_rel, alt_bins, H_sig_masked.T,
                                      cmap='magma', alpha=0.75, edgecolors='none', shading='auto')
else:
    mesh_sig = None  # nothing to plot for sig

##--Nonsignificant data (always has data)--##
H_nonsig, _, _ = np.histogram2d(time_nonsig, alt_nonsig_arr, bins=(time_bins_rel, alt_bins))
H_nonsig_percent = 100 * H_nonsig / H_nonsig.sum()
H_nonsig_masked = ma.masked_where(H_nonsig_percent == 0, H_nonsig_percent)

mesh_nonsig = ax_time_nonsig.pcolormesh(time_bins_rel, alt_bins, H_nonsig_masked.T,
                                        cmap='magma', alpha=0.75, edgecolors='none', shading='auto')

##--Format axis ticks and labels--##
for ax, has_data in [(ax_time_sig, len(time_sig) > 0 and len(alt_sig) > 0),
                     (ax_time_nonsig, True)]:  # nonsig always has data
    if has_data:
        ax.set_xlim(time_min, time_max)  # relative time axis
        ax.set_yticks(np.arange(0, 10000, 2000))
        ax.tick_params(axis='both', labelsize=12)
    else:
        ax.set_yticks(np.arange(0, 10000, 2000))
        ax.tick_params(axis='both', labelsize=12)
        ax.set_xticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")

##--Grey out any empty plots--##
if len(time_sig) == 0 or len(alt_sig) == 0:
    grey_plots(ax_map_sig)
    grey_plots(ax_time_sig)
    
    
##--Add x-axis labels--##
if len(time_sig) > 0 and len(alt_sig_arr) > 0: 
    for ax in [ax_time_sig, ax_time_nonsig]: 
        ax.set_xlabel("Days before measurement", fontsize=18)
else: 
    ax_time_nonsig.set_xlabel("Days before measurement", fontsize=18)

##--Add one colorbar, using nonsig axis which is always populated--##
cbar2 = plt.colorbar(mesh_nonsig, ax=[ax_time_sig, ax_time_nonsig],
                     orientation='vertical')
cbar2.set_label('% Trajectory Frequency', size=12)
cbar2.ax.tick_params(labelsize=14)

plt.show()
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 08:52:22 2026

@author: repooley
"""

import re
import os
import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import alphashape
from scipy.spatial import ConvexHull
from matplotlib.colors import LogNorm
from datetime import date
from pathlib import Path
from datetime import datetime

###########################
##--Establish directory--##
###########################

##--Path to this script--##
script_path = Path(__file__).resolve()

##--Path to the root which is 3 levels up in the directory--##
root = script_path.parents[3]

##--Path to raw ATom data--##
directory = root / "Arctic_NPF" / "ATom2018" / "data" / "raw" 

HYSPLIT = directory / "HYSPLIT" 

ATom = pd.read_csv(directory / "ATom.csv")

###########################################
##--Flights to be analyzed individually--##
###########################################

##--Flights to analyze - flights 2, 10, 11, 12--##
flights_to_analyze = ["Flight2"]

##################
##--Pull Files--##
##################

##--Create directory based on selected flight--##
def find_files(directory, flight):
    flight_dir = os.path.join(directory, flight)
    return flight_dir

for flight in flights_to_analyze: 
    
    print(f"Processing {flight}")
    
    #########################################################
    ##--Assign date to flight number and set up plot dims--##
    #########################################################
    
    ##--Flight-by-flight parameters--##
    if flight == "Flight2":
        flight_date = date(2018, 4, 27)
        map_extent = [-60, 180, 20, 90]
        hspace = 0.1
        htitle = 0.92
    elif flight == "Flight10":
        flight_date = date(2018, 5, 17)
        map_extent = [-180, 60, 30, 90]
        hspace = -0.15
        htitle=0.85
    elif flight == "Flight11":
        flight_date = date(2018, 5, 18)
        map_extent = [-180, 120, 30, 90]
        hspace = -0.15
        htitle=0.85
    elif flight == "Flight12":
        flight_date = date(2018, 5, 19)
        map_extent = [-180, 120, 30, 90]
        hspace = -0.15
        htitle=0.85    
    
    ##--Separate the axes from the figure object to apply different projections--##
    fig = plt.figure(figsize=(12, 10))
    
    ##--Set the height ratio between the map and curtain plot below as 2:1--##
    height_ratios = [2, 1]
    
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
    ax_time_sig.set_ylabel("Meters Above Sea Level", fontsize=12)
    #ax_time_nonsig.set_ylabel("Meters Above Ground Level", fontsize=12)
    
    fig.suptitle(f"ATom {flight.replace('Flight', 'Flight ')} ({flight_date}) HYSPLIT Back Trajectories", 
                     fontsize=16, y=htitle)

    ##########################
    ##--Loop through files--##
    ##########################
    
    lats_sig, lons_sig = [], []
    lats_nonsig, lons_nonsig = [], []
    
    magl_sig, time_sig = [], []
    magl_nonsig, time_nonsig = [], []
    p_sig, p_nonsig = [], []
    masl_sig, masl_nonsig = [], []
    
    flight_directory = find_files(HYSPLIT, flight)
    
    ##--Get timestamps where trajectories were initialized--##
    ##--Trajectories were initialized every 5 minutes from the Netcare file--##
    ##--Subset to this flight only--##
    flight_df = ATom[ATom['Flight_num'] == flight].copy()
    
    files = sorted([f for f in os.listdir(flight_directory) if f.endswith("_output.txt")])

    flight_df = ATom[ATom['Flight_num'] == flight].copy()
    flight_df['datetime'] = pd.to_datetime(flight_df['datetime'])
    
    for file in files:
        
        ##--Pull the values after the first separator (seconds since midnight--##)
        sec_str = file.split('_')[1]
        date_str = file[:8]
        
        ##--NEED to match the datetime instead to deal with dates past midnight--##
        UTC = int(sec_str)
        
        date = datetime.strptime(date_str, "%Y%m%d")
        
        ##--Create a series of date times--##
        flight_datetime = pd.to_datetime(UTC, unit='s', origin=date)
        
        match = flight_df.loc[flight_df['datetime'] == flight_datetime]

        if match.empty:
            print(f"No match for {UTC}")
            continue
        
        row = match.iloc[0]
        is_significant = pd.notna(row['nuc_significant'])
        
        ##--Determine significance--##
        #is_significant = row['nuc_significant'].notna()
        ax_map = ax_map_sig if is_significant else ax_map_nonsig
        ax_time = ax_time_sig if is_significant else ax_time_nonsig
        
        ##--For some reason, some ATom HYSPLIT files are missing a header--##
        ##--Assign the headers as follows: --##
        columns = ["TRAJ", "MGRID", "YEAR", "MONTH", "DATE", "HOUR", "MIN",
            "FOREH", "AGE", "LAT", "LONG", "ALTITUDE",
            "PRESSURE", "THETA", "AIR_TEMP", "RAINFALL",
            "MIXDEPTH", "RELHUMID", "TERR_MSL", "SUN_FLUX"]
        
        ##--Flight11 does have the headers--##
        if flight == "Flight11":
            df = pd.read_csv(os.path.join(flight_directory, file),
                sep=r'\s+')
        else: 
            df = pd.read_csv(os.path.join(flight_directory, file),
                sep=r'\s+',
                header=None,
                skiprows=1)
            
            df.columns = columns[:df.shape[1]]
        
        ##--Basic cleaning--##
        if df.empty or 'YEAR' not in df.columns:
            print(f"Skipping bad file: {file}")
            continue
        
        df = df.rename(columns={'DATE': 'DAY'})
        df['YEAR'] = df['YEAR'] + 2000
        
        df['DateTime'] = pd.to_datetime({
            'year': df['YEAR'],
            'month': df['MONTH'],
            'day': df['DAY'],
            'hour': df['HOUR']
        })
      
        ##--Group by TRAJ to place each trajectory in time order--##
        for traj_num, group in df.groupby('TRAJ'):
            group = group.sort_values('DateTime')
            
            ##--Suggestion from GPT5 - deal with HYSPLIT wrapping around meridian--##
            # Normalize to -180 to 180 range
            group['LONG'] = ((group['LONG'] + 180) % 360) - 180
            
            lon = group['LONG'].values
            lat = group['LAT'].values
            magl = group['ALTITUDE'].values
            pressure = group['PRESSURE'].values
            
            ##--Calculate meters above sea level from elev and terrain height--##
            masl = group['ALTITUDE'].values + group['TERR_MSL'].values
            
            ##--Compute relative time in days (backward from initialization)--##
            t0 = group['DateTime'].iloc[-1]
            time_rel = (group['DateTime'] - t0).dt.total_seconds() / 86400.0
            time_rel = time_rel.values  # ensure numpy array
        
            ##--Detect jumps >180° and break line by inserting NaNs--##
            jump_indices = np.where(np.abs(np.diff(lon)) > 180)[0]
            if len(jump_indices) > 0:
                for j in jump_indices[::-1]:  # reverse order to avoid index shift
                    lon = np.insert(lon, j + 1, np.nan)
                    lat = np.insert(lat, j + 1, np.nan)
                    magl = np.insert(magl, j + 1, np.nan)
                    time_rel = np.insert(time_rel, j + 1, np.nan)
                    masl = np.insert(masl, j + 1, np.nan)
            
            ##--Cut off trajectory within 1m of surface, HYSPLIT is iffy here--##
            if any(magl < 1):
                index_end = np.min(np.where(magl < 1))
            else:
                index_end = len(group) 
                
            if any(masl < 1):
                index_end = np.min(np.where(masl < 1))
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
            
            '''
            ax_map.plot(lon[:index_end],
                        lat[:index_end],
                        transform=ccrs.PlateCarree(),
                        c=color, lw=linewidth, alpha=alpha, zorder=zorder)
            '''
            ax_time.plot(time_rel[:index_end],
                         masl[:index_end],
                         c=color, lw=linewidth, alpha=alpha, zorder=zorder)
            
            ##--Append trajectory data for later histogramming--##
            if is_significant:
                lats_sig.extend(lat)
                lons_sig.extend(lon)
                magl_sig.extend(magl)
                time_sig.extend(time_rel)
                p_sig.extend(pressure)
                masl_sig.extend(masl)
            else:
                lats_nonsig.extend(lat)
                lons_nonsig.extend(lon)
                magl_nonsig.extend(magl)
                time_nonsig.extend(time_rel)  
                p_nonsig.extend(pressure)
                masl_nonsig.extend(masl)
    
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
    alpha = 0.5
    
    ##--Set min/max latitudes for binning in degrees--##
    lat_min = 30    
    lat_max = 90  
    
    ##--Set number of lon/lat bins--##
    num_theta = 50  # longitude bins
    num_r = 50       # latitude bins      
    
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
           norm=LogNorm(vmin=0.001, vmax=2),
            alpha=0.5, edgecolors='none', transform=ccrs.PlateCarree(), zorder=4)
      
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
                ax_map_sig.fill(x, y, facecolor='None', edgecolor='orangered', ls='-', lw=2,
                               transform=ccrs.PlateCarree(), zorder=5)
       
            elif hull_sig_bins.geom_type == "MultiPolygon":
                for poly in hull_sig_bins.geoms:
                    if not poly.is_empty:
                        x, y = poly.exterior.xy
                        ax_map_sig.fill(x, y, facecolor='None', edgecolor='orangered', ls='-', lw=2,
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
        norm=LogNorm(vmin=0.001, vmax=2), alpha=0.5, edgecolors='none', 
        transform=ccrs.PlateCarree(), zorder=4)
    
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
        ax_map_nonsig.fill(x, y, facecolor='None', edgecolor='orangered', ls='-',
                           linewidth=3, transform=ccrs.PlateCarree(), zorder=7)
    
    elif hull_nonsig_bins.geom_type == "MultiPolygon":
        for poly in hull_nonsig_bins.geoms:
            if not poly.is_empty:
                x, y = poly.exterior.xy
                ax_map_nonsig.fill(x, y, facecolor='None', edgecolor='orangered', ls='-', 
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
    
    num_time_bins = 12
    num_alt_bins = 10
    
    ##--Convert altitude lists to arrays--##
    alt_sig_arr = np.asarray(masl_sig)
    alt_nonsig_arr = np.asarray(masl_nonsig)
    
    ##--And time--##
    time_sig = np.asarray(time_sig)
    time_nonsig = np.asarray(time_nonsig)
    
    ##--Ensure no NaN values in any array--##
    def clean_valid_pairs(time_arr, alt_arr):
        mask = np.isfinite(time_arr) & np.isfinite(alt_arr)
        return time_arr[mask], alt_arr[mask]
    
    time_sig, alt_sig_arr = clean_valid_pairs(time_sig, alt_sig_arr)
    time_nonsig, alt_nonsig_arr = clean_valid_pairs(time_nonsig, alt_nonsig_arr)
    
    
    if len(time_sig) > 0:
        all_time_rel = np.concatenate([time_sig, time_nonsig])
    else:
        all_time_rel = np.array(time_nonsig)
    time_min = np.nanmin(all_time_rel) if len(all_time_rel) > 0 else -5  # fallback
    time_max = 0  # measurement time = 0 days
    
    ##--Create bin edges in the time dimension--##
    time_bins_rel = np.linspace(time_min, time_max, num_time_bins + 1)
    
    
    ##--Create altitude bin edges--##
    all_alt = np.concatenate([alt_sig_arr, alt_nonsig_arr]) if len(alt_sig_arr) > 0 else alt_nonsig_arr
    min_alt = np.nanmin(all_alt) if len(all_alt) > 0 else 0
    max_alt = np.nanmax(all_alt) if len(all_alt) > 0 else 10000
    alt_bins = np.linspace(min_alt, max_alt, num_alt_bins + 1)
    
    ############################
    ##--Compute 2D histogram--##
    ############################
    
    ##--Significant trajectories--##
    if len(time_sig) > 0 and len(alt_sig_arr) > 0:
        H_sig, _, _ = np.histogram2d(time_sig, alt_sig_arr, bins=(time_bins_rel, alt_bins))
        H_sig_sum = H_sig.sum()
        if H_sig_sum > 0:
            H_sig_percent = 100 * H_sig / H_sig_sum
            H_sig_masked = ma.masked_where(H_sig_percent == 0, H_sig_percent)
        else:
            H_sig_masked = ma.masked_all((len(time_bins_rel) - 1, len(alt_bins) - 1))
    
        mesh_sig = ax_time_sig.pcolormesh(time_bins_rel, alt_bins, H_sig_masked.T,
                                          cmap='magma', alpha=0.75, edgecolors='none', shading='auto')
    else:
        # Empty case
        H_sig_masked = ma.masked_all((len(time_bins_rel) - 1, len(alt_bins) - 1))
        mesh_sig = ax_time_sig.pcolormesh(time_bins_rel, alt_bins, H_sig_masked.T,
                                          cmap='Greys', alpha=0.3, shading='auto')
        grey_plots(ax_map_sig)
        grey_plots(ax_time_sig)
    
    ##--Nonsignificant trajectories--##
    if len(time_nonsig) > 0 and len(alt_nonsig_arr) > 0:
        H_nonsig, _, _ = np.histogram2d(time_nonsig, alt_nonsig_arr, bins=(time_bins_rel, alt_bins))
        H_nonsig_sum = H_nonsig.sum()
        if H_nonsig_sum > 0:
            H_nonsig_percent = 100 * H_nonsig / H_nonsig_sum
            H_nonsig_masked = ma.masked_where(H_nonsig_percent == 0, H_nonsig_percent)
        else:
            H_nonsig_masked = ma.masked_all((len(time_bins_rel) - 1, len(alt_bins) - 1))
    else:
        H_nonsig_masked = ma.masked_all((len(time_bins_rel) - 1, len(alt_bins) - 1))
    
    mesh_nonsig = ax_time_nonsig.pcolormesh(time_bins_rel, alt_bins, H_nonsig_masked.T,
                                            cmap='magma', alpha=0.75, edgecolors='none', shading='auto')
     
    ##--Format axis ticks and labels--##
    for ax, has_data in [(ax_time_sig, len(time_sig) > 0 and len(masl_sig) > 0),
                         (ax_time_nonsig, len(time_nonsig) > 0 and len(masl_nonsig) > 0)]:
        ax.set_xlim(time_min, time_max)
        ax.set_yticks(np.arange(0, 11000, 2000))
        ax.tick_params(axis='both', labelsize=12)
        if not has_data:
            ax.set_xticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")
    
    ##--Add x-axis labels--##
    if len(time_sig) > 0 and len(alt_sig_arr) > 0: 
        for ax in [ax_time_sig, ax_time_nonsig]: 
            ax.set_xlabel("Days before measurement", fontsize=18)
            
    else: 
        ax_time_nonsig.set_xlabel("Days before measurement", fontsize=18)
    
    ##--Add colorbar (based on nonsig panel)--##
    cbar2 = plt.colorbar(mesh_nonsig, ax=[ax_time_sig, ax_time_nonsig], orientation='vertical')
    cbar2.set_label('% Trajectory Frequency', size=12)
    cbar2.ax.tick_params(labelsize=14)
    
    ####################
    ##--Save figures--##
    ####################
    
    ##--Set the output directory--##
    output_dir = root / "ATom2018" / "data" / "processed"
    
    ##--Establish path to save folder--##
    folder_path = os.path.join(output_dir, "HYSPLIT", f"{flight}")
    
    ##--Create folder if it doesn't already exist--##
    os.makedirs(folder_path, exist_ok=True)
    
    ##--Use f-string to save file with flight# appended--##
    output_path = f"{folder_path}\\GriddedExtent_{flight}"
    
    ##--Save the figure--##
    plt.savefig(output_path, dpi=300, bbox_inches='tight') 
    
    plt.show()
    

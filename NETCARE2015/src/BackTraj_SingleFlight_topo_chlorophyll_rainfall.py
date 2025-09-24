# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 09:36:13 2025

@author: repooley
"""

##--This script was written referencing @eli's work--##

import os
import glob
import numpy as np
import numpy.ma as ma
import pandas as pd
import xarray as xr # to read in .nc files
import rioxarray as rio # use to downsample geospatial data
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter
from matplotlib.dates import DateFormatter, date2num, num2date
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LogNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.spatial import ConvexHull

###################
##--User inputs--##
###################

##--Set the base directory to project folder--##
directory = r"C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data"

##--HYSPLIT directory--##
hysplit = r"C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data\raw\HYSPLIT\data\trajectories"
 
##--Select flight (Flight1 thru Flight10)--##
flight = "Flight2" 

##--Base output path for figures in directory--##
output_path = r"C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data\processed\HYSPLIT"

##--Read in file containing all co-occuring data with back trajectories--##
Netcare = pd.read_csv(r"C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data\raw\Netcare.csv")

##--0.5 deg topography data from TEMIS 2010: www.temis.nl/data/gmted2010/index.php--##
topography = xr.open_dataset(r"C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data\raw\GMTED2010_15n030_0125deg.nc")

##################
##--Pull Files--##
##################

##--Define function that finds all flights available--##
##--Create directory based on selected flight--##
def find_flights(directory, flight):
    flight_dir = os.path.join(hysplit, flight)
    return flight_dir

flight_directory = find_flights(hysplit, flight)

##--Define function that creates datasets from filenames--##
def find_files(directory, flight, folder, partial_name):
    ##--flight data are stored in a folder called "raw"--##
    search_pattern = os.path.join(directory, "raw", flight, folder, f"*{partial_name}*")
    return sorted(glob.glob(search_pattern))

##--Daily chorophyll data from Aqua MODIS satellite accessed via NASA EarthData search--##
##--Right now there are three files for each and the last file is the last day before uplift--##
chlorophyll_file = find_files(directory, flight, "Chlorophyll", "AQUA")[2]
    
##--Mixing depth data from ECCO accessed via NASA EarthData search--##
mixdepth_file = find_files(directory, flight, "OceanMixDepth", "OCEAN")[2]

##--Pull chlorophyll and mixdepth data from files--##
chlorophyll = xr.open_dataset(chlorophyll_file)
mixdepth= xr.open_dataset(mixdepth_file)

##--Get timestamps where trajectories were initialized--##
##--Trajectories were initialized every 10 minutes from the Netcare file--##
single_flight = Netcare[Netcare['Flight_num'] == flight]

start_utc = int(single_flight['Time_start'].min())
end_utc = int(single_flight['Time_start'].max())
UTCs = list(range(start_utc, end_utc +1, 600))

##--Subset Netcare to times in UTCs--##
netcare_subset = single_flight[single_flight['Time_start'].isin(UTCs)]

##--Pull topography data--##
top_lon = topography['longitude']
top_lat = topography['latitude']
elevation = topography['elevation']

#############################################
##--Calculate Nanophytoplankton abundance--##
#############################################

##--See onlinelibrary.wiley.com/doi/abs/10.1029/2005JC003207 for method--##

##--Pull chlor-a and mixing depth variables--##
chlor_a = chlorophyll['chlor_a']

##--Pull mixing depth--##
mixing_depth = mixdepth["MXLDEPTH"]

##--Squeeze out first unneeded time dimension from mixing_depth--##
mixing_depth = mixing_depth.squeeze('time')

##--Reproject the chlorophyll data to resolution of mixing depth data--##

##--First specify the CRS of both datasets--##
chlor_a.rio.write_crs("EPSG:4326", inplace=True)
mixing_depth.rio.write_crs("EPSG:4326", inplace=True)
chlor_reproj = chlor_a.rio.reproject_match(mixing_depth)

##--Pull lat/lon from mixdepth--##
chlor_lon = mixing_depth['longitude']
chlor_lat = mixing_depth['latitude']

##--Calculate the ocean euphotic layer depth using surface Chlor-a--##
##--Fit calculated in Excel, file is with the raw chlorophyll data--##
euphotic_depth = -22.73 * np.log(0.006023* chlor_reproj)

##--Estimate the mixing state by comparing Zeu to the mixed-layer depth--##
mixing_state = xr.DataArray(euphotic_depth.data / mixing_depth.data,
    dims=euphotic_depth.dims, coords=euphotic_depth.coords)

##--Assign trophic level to each data point based on chlorophyll abundance--##
stratified_levels = { "S1": (0.0, 0.047), "S2": (0.048, 0.080), "S3": (0.081, 0.118),
    "S4": (0.119, 0.162), "S5": (0.163, 0.296), "S6": (0.297, 0.444), 
    "S7": (0.445, 0.888), "S8": (0.889, 2.094), "S9": (2.095, np.inf)}

mixed_levels = {"M1": (0.0, 0.414), "M2": (0.415, 0.742), "M3": (0.743, 1.216),
    "M4": (1.217, 4.752), "M5": (4.753, np.inf)}

##--Assistance from GPT-5 model for writing vectorized function--##
##--Inputs: chlorophyll, mixing state--##
def assign_trophic(chlor, mixing):
    
    ##--Handle NaNs--##
    if np.isnan(chlor) or np.isnan(mixing):
        return np.nan
    
    ##--First sort by mixing state--##
    if mixing >= 1:  # Stratified case
    
        ##--Compare 
        for level, (low, high) in stratified_levels.items():
            if low <= chlor < high:
                return level
        
    else:  # Mixed case
        for level, (low, high) in mixed_levels.items():
            if low <= chlor < high:
                return level
            
    ##--Return NaN for anything that doesn't categorize--##
    return np.nan

##--Vectorize function within apply_ufunc--##
trophic_levels = xr.apply_ufunc(
    np.vectorize(assign_trophic), chlor_reproj, mixing_state,
    ##--Function with logic doesn't inherently work with dask, tells it to split into chunks--##
    input_core_dims=[[], []], vectorize=True, dask="parallelized",
    ##--Output type is a string, eg "S1"--##
    output_dtypes=[str])

##--Assign percent of nanophytoplankton for each mixing/trophic level--##
nano_percent_strat = {"S1": 0.44, "S2": 0.418, "S3": 0.421, "S4": 0.445,
    "S5": 0.493, "S6": 0.483, "S7": 0.452, "S8": 0.328, "S9": 0.211}

nano_percent_mixed = {"M1": 0.507, "M2": 0.498, "M3": 0.572,
    "M4": 0.381, "M5": 0.053}

##--Determine nanophytoplankton abundance based on trophic assignment--##
def compute_nano(chlor, trophic):
    ##--Handle NaNs--##
    if np.isnan(chlor) or trophic is None or trophic == "nan":
        return np.nan
    
    ##--Assign mixed/strat based on trophic level starting letter--##
    if trophic.startswith("S"):
        ##--Get the value from the key and assign as 'percent'--##
        percent = nano_percent_strat.get(trophic, np.nan)
    elif trophic.startswith("M"):
        percent = nano_percent_mixed.get(trophic, np.nan)
        
    else:
        return np.nan
    
    ##--Multiply chorophyll by assigned percent based on trophic level--##
    return chlor * percent

##--Vectorize function--##
nano_abundance = xr.apply_ufunc(np.vectorize(compute_nano), chlor_reproj,
    trophic_levels, input_core_dims=[[], []], vectorize=True,
    dask="parallelized", output_dtypes=[float])

##--Drop zeroes from chlorophyll data to use LogNorm scale--##
nano_abundance = np.where(nano_abundance > 0, nano_abundance, np.nan)

###################
##--Set up plot--##
###################

##--Separate the axes from the figure object to apply different projections--##
fig = plt.figure(figsize=(12, 12), constrained_layout=True)

##--Use gridspec to access figure layout--##
gs = gridspec.GridSpec(nrows=4, ncols=2, height_ratios=[2.5, 1, 1, 1], figure=fig)

##--Create a map with polar stereographic projection in first subplot--##
##--First column: NPF is significant--##
ax_map_sig = fig.add_subplot(gs[0, 0], projection=ccrs.NorthPolarStereo())
ax_temp_sig = fig.add_subplot(gs[1, 0])
ax_RH_sig = fig.add_subplot(gs[2, 0])
ax_rain_sig = fig.add_subplot(gs[3, 0])

##--Second column: non-significant NPF--##
ax_map_nonsig = fig.add_subplot(gs[0, 1], projection=ccrs.NorthPolarStereo())
ax_temp_nonsig = fig.add_subplot(gs[1, 1])
ax_RH_nonsig = fig.add_subplot(gs[2, 1])
ax_rain_nonsig = fig.add_subplot(gs[3, 1])

##--Get topography cmap and set low elevation to no color--##
colors = ['#355E3B', '#FFFDD0'] # hunter green and cream
cmap = LinearSegmentedColormap.from_list('TwoColorGradient', colors, N=256)
cmap.set_under('none') # transparent
##--Normalize the cmap, transparent under 1 m elev--##
norm = plt.Normalize(vmin=1, vmax=6000)

##--Get chlorophyll cmap (zeros are ocean colored)--##
colors2 = ['#F0FFFF', '#001A13'] # pale blue and deep Kelly green
cmap2 = LinearSegmentedColormap.from_list('TwoColorGradient', colors2, N=256)

##--Basic map formatting--##
for ax_map in [ax_map_sig, ax_map_nonsig]:
    if flight == 'Flight2':
        ax_map.set_extent([-120, 10, 38, 85], crs=ccrs.PlateCarree())
    else: 
        ax_map.set_extent([-140, 140, 40, 85], crs=ccrs.PlateCarree())
    
    ##--Add ocean layer to maps--##
    ax_map.add_feature(cfeature.OCEAN, fc='#F0FFFF', ec='k', lw=0.2, zorder=1)

    ##--Add topographic data to maps--##
    elev_map = ax_map.pcolormesh(top_lon, top_lat, elevation, 
        transform=ccrs.PlateCarree(), # match transform to projection used
        cmap=cmap, norm=norm, shading='auto', zorder=1) 

    ##--Add nanophytoplankton data to maps--##
    chlor_map = ax_map.pcolormesh(chlor_lon, chlor_lat, nano_abundance,
          transform=ccrs.PlateCarree(), cmap=cmap2, 
          norm=LogNorm(vmin=0.01, vmax=100), shading='auto', zorder=1)

##--Set labels--##
ax_map_sig.set_title("$N_{2.5-10}$ Event", fontsize=24)
ax_map_nonsig.set_title("No Significant $N_{2.5-10}$", fontsize=24)
ax_RH_sig.set_ylabel("Meters Above Ground Level", fontsize=20)
fig.suptitle(f"NETCARE {flight.replace('Flight', 'Flight ')} HYSPLIT Back Trajectories", fontsize=30, y=1.05)

##--Set axes limits--##
ax_temp_sig.set_ylim(-250, 8000)
ax_temp_nonsig.set_ylim(-250, 8000)
ax_RH_sig.set_ylim(-250, 8000)
ax_RH_nonsig.set_ylim(-250, 8000)
ax_rain_sig.set_ylim(-250, 8000)
ax_rain_nonsig.set_ylim(-250, 8000)

##--Map colorbars--##
cax1 = fig.add_axes([0.92, 0.78, 0.01, 0.17])   # [left, bottom, width, height]
cb1 = fig.colorbar(elev_map, cax=cax1, orientation="vertical", shrink=0.1)
cb1.set_label("Elevation (m)", size=14)
cb1.ax.tick_params(labelsize=14)

cax2 = fig.add_axes([0.92, 0.60, 0.01, 0.17])   # same x position, lower bottom
cb2 = fig.colorbar(chlor_map, cax=cax2, orientation="vertical", shrink=0.1)
cb2.set_label(r"1-day Nano$_{\rm surf}$ (mg/m$^3$)", size=14)
cb2.ax.tick_params(labelsize=14)

##########################
##--Group trajectories--##
##########################

##--Sort trajectory outputs into signficant or non-significant NPF lists--##
lats_sig, lats_nonsig = [], []
lons_sig, lons_nonsig = [], []
alt_sig, alt_nonsig = [], []
time_sig, time_nonsig = [], []
temp_sig, temp_nonsig = [], []
RH_sig, RH_nonsig = [], []
rain_sig, rain_nonsig = [], []

for file, row in zip(sorted(os.listdir(flight_directory)), netcare_subset.itertuples(index=False)):
    
    ##--Condition: initialized traj must be above the marginal polar dome--##
    ##--Marginal dome boundary is 285 K--##
    if row.ptemp > 285:

        ##--Determine which axis to use (NPF vs non-NPF)--##
        is_significant = pd.notna(row.nuc_significant)
        ax_map = ax_map_sig if is_significant else ax_map_nonsig
        ax_temp = ax_temp_sig if is_significant else ax_temp_nonsig
        ax_RH = ax_RH_sig if is_significant else ax_RH_nonsig
        ax_rain = ax_rain_sig if is_significant else ax_rain_nonsig
            
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
    
            ##--Cut off trajectory within 1m of surface, HYSPLIT is iffy here--##
            if any(altitudes < 1):
                index_end = np.min(np.where(altitudes < 1))
            else:
                index_end = len(group) 
                
           ##--Plot original (unperturbed) trajectory--##
            if traj_num == 1:
                color = 'k'
                linewidth = 1.25
                alpha = 1
                zorder = 5
            else:
                color = 'none'
            
            ax_map.plot(group['LONG'].iloc[:index_end],
                group['LAT'].iloc[:index_end],
                transform=ccrs.PlateCarree(),
                c=color, lw=linewidth, alpha=alpha, zorder=zorder)
            
            ax_temp.plot(group['DateTime'].iloc[:index_end],
                group['ALTITUDE'].iloc[:index_end],
                c=color, lw=linewidth, alpha=alpha, zorder=zorder)
            
            ax_RH.plot(group['DateTime'].iloc[:index_end],
                group['ALTITUDE'].iloc[:index_end],
                c=color, lw=linewidth, alpha=alpha, zorder=zorder)
            
            ax_rain.plot(group['DateTime'].iloc[:index_end],
                group['ALTITUDE'].iloc[:index_end],
                c=color, lw=linewidth, alpha=alpha, zorder=zorder)
            
            if is_significant:
                lats_sig.extend(group['LAT'].values)
                lons_sig.extend(group['LONG'].values)
                alt_sig.extend(group['ALTITUDE'].values)
                time_sig.extend(group['DateTime'].tolist())
                temp_sig.extend(group['AIR_TEMP'].tolist())
                RH_sig.extend(group['RELHUMID'].values)
                rain_sig.extend(group['RAINFALL'].values)
            else:
                lats_nonsig.extend(group['LAT'].values)
                lons_nonsig.extend(group['LONG'].values)
                alt_nonsig.extend(group['ALTITUDE'].values)
                time_nonsig.extend(group['DateTime'].tolist())
                temp_nonsig.extend(group['AIR_TEMP'].tolist())
                RH_nonsig.extend(group['RELHUMID'].values)
                rain_nonsig.extend(group['RAINFALL'].values)

##--Add colorbars for elevation and chlorophyll a--##

############################################
##--Map: Convex Hull around trajectories--##
############################################

##--First, create a histogram for the trajectories--##

##--Set min/max latitudes for binning in degrees--##
lat_min = 30    
lat_max = 90  

##--Set number of lon/lat bins in radians--##
num_theta = 60  # longitude bins
num_r = 20      # latitude bins  

##--Calculate longitude edges and create bins--##
lon_edges = np.linspace(-180, 180, num_theta + 1)
theta_bins = np.deg2rad(lon_edges)

##--From GPT-5 model: Make latitude bin sizes uniform by taking sine of radians--##
sin_edges = np.linspace(np.sin(np.deg2rad(lat_min)), np.sin(np.deg2rad(lat_max)), num_r + 1)
lat_edges = np.rad2deg(np.arcsin(sin_edges))

##--Convert bins to polar radius and sort in ascending order--##
r_bins = np.sort(90 - lat_edges) 

##--Convert degree lat/lons to radians--##
##--Separate by sig/nonsig NPF--##
theta_sig = np.deg2rad(np.array(lons_sig) + 180)
theta_nonsig = np.deg2rad(np.array(lons_nonsig) + 180)

r_sig = 90 - np.abs(np.array(lats_sig))
r_nonsig = 90 - np.abs(np.array(lats_nonsig))

##--Mask to only include data below 1000m altitude--##
##--Convert altitudes to arrays from lists--##
alt_sig_arr = np.array(alt_sig)
alt_nonsig_arr = np.array(alt_nonsig)

##--Create the mask--##
alt_sig_mask = alt_sig_arr < 500
alt_nonsig_mask = alt_nonsig_arr < 500

##--Apply mask to theta and r dimensions--##
theta_sig_alt = theta_sig[alt_sig_mask]
r_sig_alt = r_sig[alt_sig_mask]

theta_nonsig_alt = theta_nonsig[alt_nonsig_mask]
r_nonsig_alt = r_nonsig[alt_nonsig_mask]

##--Create 2d histograms--##
H_sig, _, _ = np.histogram2d(theta_sig_alt, r_sig_alt, bins=(theta_bins, r_bins))
H_nonsig, _, _ = np.histogram2d(theta_nonsig_alt, r_nonsig_alt, bins=(theta_bins, r_bins))

##--Compute percent frequency for each bin--##
H_sig_percent = 100 * H_sig / H_sig.sum()
H_nonsig_percent = 100 * H_nonsig / H_nonsig.sum()

##--Mask bins with zero frequency--##
H_sig_masked = ma.masked_where(H_sig_percent == 0, H_sig_percent)
H_nonsig_masked = ma.masked_where(H_nonsig_percent == 0, H_nonsig_percent)

##--Create a meshgrid for plotting with pcolormesh--##
theta_grid, r_grid = np.meshgrid(theta_bins, r_bins, indexing='ij')
lon_grid = np.rad2deg(theta_grid) - 180
lat_grid = 90 - r_grid

##--Flatten the histograms--##
flat_sig = H_sig_percent.flatten()
flat_nonsig = H_nonsig_percent.flatten()

##--Sort bins by frequency--##
##--Sort by percent, not xy location--##
sorted_sig_idx = np.argsort(flat_sig)[::-1]  # descending order
##--Cumulate all bins--##
cum_sig = np.cumsum(flat_sig[sorted_sig_idx])

sorted_nonsig_idx = np.argsort(flat_nonsig)[::-1]
cum_nonsig = np.cumsum(flat_nonsig[sorted_nonsig_idx])

##--Establish lower cutoff for bins with lower 10% of data based on index--##
cutoff_sig_idx = np.where(cum_sig <= 90)[0] 
keep_sig_idx = sorted_sig_idx[cutoff_sig_idx]

cutoff_nonsig_idx = np.where(cum_nonsig <= 90)[0]
keep_nonsig_idx = sorted_nonsig_idx[cutoff_nonsig_idx]

##--Create mask with selected bins above cutoff--##
mask_sig_top90 = np.zeros_like(flat_sig, dtype=bool) # establish zeroes
mask_sig_top90[keep_sig_idx] = True # index the mask
mask_sig_top90 = mask_sig_top90.reshape(H_sig_percent.shape) # shape the mask

mask_nonsig_top90 = np.zeros_like(flat_nonsig, dtype=bool)
mask_nonsig_top90[keep_nonsig_idx] = True
mask_nonsig_top90 = mask_nonsig_top90.reshape(H_nonsig_percent.shape)

##--Further mask to only include data below 1000 m altitude--##


selected_sig_boxes = []
for i in range(mask_sig_top90.shape[0]):
    for j in range(mask_sig_top90.shape[1]):
        if mask_sig_top90[i, j]:
            lon_center = 0.25 * (lon_grid[i, j] + lon_grid[i+1, j] +
                                 lon_grid[i+1, j+1] + lon_grid[i, j+1])
            lat_center = 0.25 * (lat_grid[i, j] + lat_grid[i+1, j] +
                                 lat_grid[i+1, j+1] + lat_grid[i, j+1])
            selected_sig_boxes.append((lon_center, lat_center))
                
selected_nonsig_boxes = []
for i in range(mask_nonsig_top90.shape[0]):  
    for j in range(mask_nonsig_top90.shape[1]):  
        if mask_nonsig_top90[i, j]:
            lon_corners = [lon_grid[i, j], lon_grid[i+1, j], lon_grid[i+1, j+1], lon_grid[i, j+1]]
            lat_corners = [lat_grid[i, j], lat_grid[i+1, j], lat_grid[i+1, j+1], lat_grid[i, j+1]]
            for lon, lat in zip(lon_corners, lat_corners):
                selected_nonsig_boxes.append((lon, lat))

points_sig_bins = np.unique(selected_sig_boxes, axis=0)
points_nonsig_bins = np.unique(selected_nonsig_boxes, axis=0)
hull_sig_bins = ConvexHull(points_sig_bins)
hull_nonsig_bins = ConvexHull(points_nonsig_bins)

# Plot hull polygon
for simplex in hull_sig_bins.simplices:
    ax_map_sig.plot(points_sig_bins[simplex, 0], points_sig_bins[simplex, 1],
                    c='#9D0759', ls='--', lw=2,
                    transform=ccrs.PlateCarree(), zorder=5)
    
for simplex in hull_nonsig_bins.simplices:
    ax_map_nonsig.plot(points_nonsig_bins[simplex, 0], points_nonsig_bins[simplex, 1],
                    c='#9D0759', ls='--', lw=2,
                    transform=ccrs.PlateCarree(), zorder=5)

########################
##--Histogram set up--##
########################

##--These bin numbers apply to ALL curtain plots--##
num_time_bins = 10
num_alt_bins = 8

##--Convert all datetime objects to float days since 1970--##
time_sig_num = date2num(time_sig)
time_nonsig_num = date2num(time_nonsig)

# Combine for bin range
all_time_num = np.concatenate([time_sig_num, time_nonsig_num])
time_bins_num = np.linspace(all_time_num.min(), all_time_num.max(), num_time_bins + 1)

alt_sig_arr = np.array(alt_sig)
alt_nonsig_arr = np.array(alt_nonsig)
alt_bins = np.linspace(min(alt_sig_arr.min(), alt_nonsig_arr.min()),
                       max(alt_sig_arr.max(), alt_nonsig_arr.max()),
                       num_alt_bins + 1)

##--Count the number of datapoints in each bin--##
sig_count, _, _ = np.histogram2d(time_sig_num, alt_sig_arr, bins=(time_bins_num, alt_bins))
nonsig_count, _, _ = np.histogram2d(time_nonsig_num, alt_nonsig, bins=(time_bins_num, alt_bins))

for ax in [ax_temp_sig, ax_temp_nonsig,
           ax_RH_sig, ax_RH_nonsig,
           ax_rain_sig, ax_rain_nonsig]:
    ax.set_xlim(time_bins_num.min(), time_bins_num.max())  # set range explicitly
    ax.xaxis_date()  # interpret floats as dates
    ax.xaxis.set_major_formatter(DateFormatter('%b %d'))
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_horizontalalignment('right')
    ax.tick_params(axis='both', labelsize=16)

###################
##--Temperature--##
###################

##--Convert temperature lists to arrays--##
temp_sig_arr = np.array(temp_sig)
temp_nonsig_arr = np.array(temp_nonsig)

##--Sum all temperature values in each histogram bin--##
temp_sig_sum, _, _ = np.histogram2d(time_sig_num, alt_sig_arr, 
                        bins=(time_bins_num, alt_bins), weights=temp_sig_arr)
temp_nonsig_sum, _, _ = np.histogram2d(time_nonsig_num, alt_nonsig,
                        bins=(time_bins_num, alt_bins), weights=temp_nonsig_arr)

##--Compute the average RH per bins with counts--##
temp_sig_mean = np.divide(temp_sig_sum, sig_count, 
                out=np.full_like(temp_sig_sum, np.nan), where=sig_count > 0)
temp_nonsig_mean = np.divide(temp_nonsig_sum, nonsig_count, 
                out=np.full_like(temp_nonsig_sum, np.nan), where=nonsig_count > 0)

##--Plot alitude vs temp--##
mesh_sig = ax_temp_sig.pcolormesh(time_bins_num, alt_bins, temp_sig_mean.T,
                       cmap='magma', alpha=0.75, edgecolors='none', shading='auto',
                       vmin=220, vmax=300)
mesh_nonsig = ax_temp_nonsig.pcolormesh(time_bins_num, alt_bins, temp_nonsig_mean.T,
                          cmap='magma', alpha=0.75, edgecolors='none', shading='auto',
                          vmin=220, vmax=300)

##--Remove tick labels - share with bottom plots--##
for ax in [ax_temp_sig, ax_temp_nonsig]:
    ax.tick_params(axis='x', labelbottom=False)
    ax.tick_params(axis='both', labelsize=16) 

##--Add one colorbar--##
cbar2 = plt.colorbar(mesh_sig, ax=[ax_temp_sig, ax_temp_nonsig],
                    orientation='vertical', fraction=0.075, pad=0.04)
cbar2.set_label('Temperature (K)', size=14)
cbar2.ax.tick_params(labelsize=14)

##########
##--RH--##
##########

##--Convert RH lists to arrays--##
RH_sig_arr = np.array(RH_sig)
RH_nonsig_arr = np.array(RH_nonsig)

##--Sum all RH values in each histogram bin--##
RH_sig_sum, _, _ = np.histogram2d(time_sig_num, alt_sig_arr, 
                        bins=(time_bins_num, alt_bins), weights=RH_sig_arr)
RH_nonsig_sum, _, _ = np.histogram2d(time_nonsig_num, alt_nonsig,
                        bins=(time_bins_num, alt_bins), weights=RH_nonsig_arr)

##--Compute the average RH per bins with counts--##
RH_sig_mean = np.divide(RH_sig_sum, sig_count, 
                out=np.full_like(RH_sig_sum, np.nan), where=sig_count > 0)
RH_nonsig_mean = np.divide(RH_nonsig_sum, nonsig_count, 
                out=np.full_like(RH_nonsig_sum, np.nan), where=nonsig_count > 0)

##--Plot RH--##
mesh_RH_sig = ax_RH_sig.pcolormesh(time_bins_num, alt_bins, RH_sig_mean.T,
                                   cmap='viridis', alpha=0.75, shading='auto',
                                   vmin=0, vmax=105)
mesh_RH_nonsig = ax_RH_nonsig.pcolormesh(time_bins_num, alt_bins, RH_nonsig_mean.T,
                                         cmap='viridis', alpha=0.75, shading='auto',
                                         vmin=0, vmax=105)

##--Remove tick labels - share with bottom plots--##
for ax in [ax_RH_sig, ax_RH_nonsig]:
    ax.tick_params(axis='x', labelbottom=False)
    ax.tick_params(axis='both', labelsize=16) 

##--Add one colorbar--##
cbar3 = plt.colorbar(mesh_RH_sig, ax=[ax_RH_sig, ax_RH_nonsig],
                     orientation='vertical', fraction=0.075, pad=0.04)
cbar3.set_label('Relative Humidity (%)', size=14)
cbar3.ax.tick_params(labelsize=14)

################
##--Rainfall--##
################

##--Convert rainfall lists to arrays--##
rain_sig_arr = np.array(rain_sig)
rain_nonsig_arr = np.array(rain_nonsig)

##--Sum of rainfall (in mm/hr) per each bin--##
rain_sig_sum, _, _ = np.histogram2d(time_sig_num, alt_sig_arr, 
                        bins=(time_bins_num, alt_bins), weights=rain_sig_arr)
rain_nonsig_sum, _, _ = np.histogram2d(time_nonsig_num, alt_nonsig,
                        bins=(time_bins_num, alt_bins), weights=rain_nonsig_arr)

##--Average rainfall per bin for values greater than 0--##
rain_sig_mean = np.divide(rain_sig_sum, sig_count, 
                out=np.full_like(rain_sig_sum, np.nan), where=sig_count > 0)
rain_nonsig_mean = np.divide(rain_nonsig_sum, nonsig_count, 
                out=np.full_like(rain_nonsig_sum, np.nan), where=nonsig_count > 0)

##--Rainfall curtain plots--##
mesh_rain_sig = ax_rain_sig.pcolormesh(time_bins_num, alt_bins, rain_sig_mean.T,
                                   cmap='Blues', alpha=0.75, shading='auto',
                                   vmin=0, vmax=2)
mesh_rain_nonsig = ax_rain_nonsig.pcolormesh(time_bins_num, alt_bins, rain_nonsig_mean.T,
                                         cmap='Blues', alpha=0.75, shading='auto', 
                                         vmin=0, vmax=2)

##--Add one colorbar--##
cbar4 = plt.colorbar(mesh_rain_sig, ax=[ax_rain_sig, ax_rain_nonsig],
                     orientation='vertical', fraction=0.075, pad=0.04)
cbar4.set_label('Rainfall (mm/hr)', size=14)
cbar4.ax.tick_params(labelsize=14)

##--Add text labels to each set of plots--##
ax_map_sig.text(0.61, 0.93, 'Nanophytoplankton', horizontalalignment='center', 
         verticalalignment='center', transform=ax_map_sig.transAxes, fontsize=18,
         bbox=dict(boxstyle="round, pad=0.5", fc="white", ec='none', lw=1, alpha=0.75))
ax_temp_sig.text(0.80, 0.9, 'Temperature', horizontalalignment='center', 
         verticalalignment='center', transform=ax_temp_sig.transAxes, fontsize=18)
ax_RH_sig.text(0.73, 0.9, 'Relative Humidity', horizontalalignment='center', 
         verticalalignment='center', transform=ax_RH_sig.transAxes, fontsize=18)
ax_rain_sig.text(0.87, 0.9, 'Rainfall', horizontalalignment='center', 
         verticalalignment='center', transform=ax_rain_sig.transAxes, fontsize=18)

plt.show()
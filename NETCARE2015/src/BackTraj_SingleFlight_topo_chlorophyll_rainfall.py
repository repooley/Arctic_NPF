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
import cmcrameri # pretty colors
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.dates import DateFormatter, date2num, num2date
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LogNorm
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.spatial import ConvexHull
import alphashape

###################
##--User inputs--##
###################

##--Set the base directory to project folder--##
directory = r"C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data"

##--HYSPLIT directory--##
hysplit = r"C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data\raw\HYSPLIT\data\trajectories\5min_averaged"
 
##--Select flight (Flight1 thru Flight10)--##
flight = "Flight5" 

##--Filter for above polar dome?--##
above_dome = True

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
chlorophyll_file = find_files(directory, flight, "Chlorophyll", "AQUA")[0]
    
##--Mixing depth data from ECCO accessed via NASA EarthData search--##
mixdepth_file = find_files(directory, flight, "OceanMixDepth", "OCEAN")[0]

##--Pull chlorophyll and mixdepth data from files--##
chlorophyll = xr.open_dataset(chlorophyll_file)
mixdepth= xr.open_dataset(mixdepth_file)

##--Get timestamps where trajectories were initialized--##
##--Trajectories were initialized every 10 minutes from the Netcare file--##
single_flight = Netcare[Netcare['Flight_num'] == flight]

start_utc = int(single_flight['Time_start'].min())
end_utc = int(single_flight['Time_start'].max())
UTCs = list(range(start_utc, end_utc +1, 300))

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

##--Flight-by-flight parameters--##
if flight == "Flight2":
    map_extent = [-100, 10, 35, 90]
    height_ratios = [2.5, 1, 1, 1]
    width_ratios = [1, 1, 0.025]
    hspace = 0.25
    wspace=0.33
    htitle = 0.95
elif flight == "Flight3":
    map_extent = [-120, 10, 45, 90]
    height_ratios = [2.5, 1, 1, 1]
    width_ratios = [1, 1, 0.05]
    hspace = 0.2
    wspace=0.3
    htitle=0.95
elif flight == "Flight4":
    map_extent = [-120, 15, 60, 90]
    height_ratios = [2.5, 1, 1, 1]
    width_ratios = [1, 1, 0.05]
    hspace = 0.25
    wspace=0.33
    htitle=0.96
elif flight == "Flight5":
    map_extent = [-120, 15, 60, 90]
    height_ratios = [2.5, 1, 1, 1]
    width_ratios = [1, 1, 0.05]
    hspace = 0.25
    wspace=0.33
    htitle=0.96
elif flight == "Flight6":
    map_extent = [-120, 10, 65, 90]
    height_ratios = [2, 1, 1, 1]
    width_ratios = [1, 1, 0.05]
    hspace = 0.1
    wspace=0.05
    htitle=0.91
elif flight == "Flight7":
    map_extent = [-120, 32, 45, 90]
    height_ratios = [2.5, 1, 1, 1]
    width_ratios = [1, 1, 0.05]
    hspace = 0.0
    wspace=0.05
    htitle=0.85
elif flight == "Flight8":
    map_extent = [-180, 120, 28, 90]
    height_ratios = [2, 1, 1, 1]
    width_ratios = [1, 1, 0.05]
    hspace = 0.0
    htitle=0.85
elif flight == "Flight9":
    map_extent = [-180, 60, 28, 90]
    height_ratios = [2, 1, 1, 1]
    width_ratios = [1, 1, 0.05]
    hspace = -0.1
    wspace=0.05
    htitle=0.88
elif flight == "Flight10":
    map_extent = [-180, 60, 28, 90]
    height_ratios = [2, 1, 1, 1]
    width_ratios = [1, 1, 0.05]
    hspace = -0.1
    wspace=0.05
    htitle=0.88
else: 
    map_extent = [-120, 10, 70, 90]
    height_ratios = [2, 1, 1, 1]
    width_ratios = [1, 1, 0.05]
    hspace = 0.1
    wspace=0.05
    htitle=0.91
    
##--Separate the axes from the figure object to apply different projections--##
fig = plt.figure(figsize=(12, 12))

##--Use gridspec to access figure layout--##
gs = gridspec.GridSpec(4, 3, height_ratios=height_ratios, width_ratios=width_ratios, 
                      hspace=hspace, wspace=wspace)

##--Center maps along the central longitude of the map extent--##
#central_lon = (map_extent[0] + map_extent[1]) / 2

##--Create a map with polar stereographic projection in first subplot--##
##--First column: NPF is significant--##
ax_map_sig = fig.add_subplot(gs[0, 0], 
            projection=ccrs.NorthPolarStereo())
ax_temp_sig = fig.add_subplot(gs[1, 0])
ax_RH_sig = fig.add_subplot(gs[2, 0])
ax_rain_sig = fig.add_subplot(gs[3, 0])

##--Second column: non-significant NPF--##
ax_map_nonsig = fig.add_subplot(gs[0, 1], 
            projection=ccrs.NorthPolarStereo())
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
cmap2 = cmcrameri.cm.navia
cmap2.set_under('none')

##--Basic map formatting--##
for ax_map in [ax_map_sig, ax_map_nonsig]:
  
    ##--Set map extent--##
    ax_map.set_extent(map_extent, crs=ccrs.PlateCarree())
 
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
  
##--Right column colorbar slots--##
##--Suggestion from GPT 5: split map colorbar slots into two--##
cax_mapslot = fig.add_subplot(gs[0, 2])

##--Delete the dummy axis (we’ll replace it with inset axes)--##
cax_mapslot.remove()

##--Create two inset colorbars inside the map colorbar slot--##
cax_elev = inset_axes(ax_map_nonsig, width="6%", height="50%",
                      loc="center right", bbox_to_anchor=(0.25, 0.3, 1, 1),
                      bbox_transform=ax_map_nonsig.transAxes, borderpad=0)
cax_chlor = inset_axes(ax_map_nonsig, width="6%", height="50%",
                       loc="center right", bbox_to_anchor=(0.25, -0.3, 1, 1),
                       bbox_transform=ax_map_nonsig.transAxes, borderpad=0)

cax_temp  = fig.add_subplot(gs[1, 2])
cax_RH    = fig.add_subplot(gs[2, 2])
cax_rain  = fig.add_subplot(gs[3, 2])

##--Set labels--##
ax_map_sig.set_title("$N_{2.5-10}$ Event", fontsize=18)
ax_map_nonsig.set_title("No Significant $N_{2.5-10}$", fontsize=18)
ax_RH_sig.set_ylabel("Meters Above Ground Level", fontsize=20)
fig.suptitle(f"NETCARE {flight.replace('Flight', 'Flight ')} HYSPLIT Back Trajectories", 
             fontsize=24, y=htitle)

##--Set axes limits--##
ax_temp_sig.set_ylim(-250, 8000)
ax_temp_nonsig.set_ylim(-250, 8000)
ax_RH_sig.set_ylim(-250, 8000)
ax_RH_nonsig.set_ylim(-250, 8000)
ax_rain_sig.set_ylim(-250, 8000)
ax_rain_nonsig.set_ylim(-250, 8000)

##--Map colorbars--## 
cb1 = fig.colorbar(elev_map, cax=cax_elev, orientation="vertical", shrink=0.5)
cb1.set_label("Elevation (m)", size=12)
cb1.ax.tick_params(labelsize=14)

cb2 = fig.colorbar(chlor_map, cax=cax_chlor, orientation="vertical", shrink=0.5)
cb2.set_label(r"Nano$_{\rm surf}$ (mg/m$^3$)", size=12)
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
    
    if above_dome == True: 
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
                    linewidth = 1.25
                    alpha = 1
                    zorder = 5
                else:
                    color = 'none'
                
                ax_map.plot(group['LONG'].iloc[:index_end],
                    group['LAT'].iloc[:index_end],
                    transform=ccrs.PlateCarree(),
                    c=color, lw=linewidth, alpha=alpha, zorder=zorder)
                
                ax_temp.plot(time_rel.iloc[:index_end],
                    group['ALTITUDE'].iloc[:index_end],
                    c=color, lw=linewidth, alpha=alpha, zorder=zorder)
                
                ax_RH.plot(time_rel.iloc[:index_end],
                    group['ALTITUDE'].iloc[:index_end],
                    c=color, lw=linewidth, alpha=alpha, zorder=zorder)
                
                ax_rain.plot(time_rel.iloc[:index_end],
                    group['ALTITUDE'].iloc[:index_end],
                    c=color, lw=linewidth, alpha=alpha, zorder=zorder)
                
                if is_significant:
                    lats_sig.extend(group['LAT'].values)
                    lons_sig.extend(group['LONG'].values)
                    alt_sig.extend(group['ALTITUDE'].values)
                    time_sig.extend(time_rel.values)  
                    temp_sig.extend(group['AIR_TEMP'].tolist())
                    RH_sig.extend(group['RELHUMID'].values)
                    rain_sig.extend(group['RAINFALL'].values)
                else:
                    lats_nonsig.extend(group['LAT'].values)
                    lons_nonsig.extend(group['LONG'].values)
                    alt_nonsig.extend(group['ALTITUDE'].values)
                    time_nonsig.extend(time_rel.values)  
                    temp_nonsig.extend(group['AIR_TEMP'].tolist())
                    RH_nonsig.extend(group['RELHUMID'].values)
                    rain_nonsig.extend(group['RAINFALL'].values)
    else: 
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
                linewidth = 1.25
                alpha = 1
                zorder = 5
            else:
                color = 'none'
            
            ax_map.plot(group['LONG'].iloc[:index_end],
                group['LAT'].iloc[:index_end],
                transform=ccrs.PlateCarree(),
                c=color, lw=linewidth, alpha=alpha, zorder=zorder)
            
            ax_temp.plot(time_rel.iloc[:index_end],
                group['ALTITUDE'].iloc[:index_end],
                c=color, lw=linewidth, alpha=alpha, zorder=zorder)
            
            ax_RH.plot(time_rel.iloc[:index_end],
                group['ALTITUDE'].iloc[:index_end],
                c=color, lw=linewidth, alpha=alpha, zorder=zorder)
            
            ax_rain.plot(time_rel.iloc[:index_end],
                group['ALTITUDE'].iloc[:index_end],
                c=color, lw=linewidth, alpha=alpha, zorder=zorder)
            
            if is_significant:
                lats_sig.extend(group['LAT'].values)
                lons_sig.extend(group['LONG'].values)
                alt_sig.extend(group['ALTITUDE'].values)
                time_sig.extend(time_rel.values)  
                temp_sig.extend(group['AIR_TEMP'].tolist())
                RH_sig.extend(group['RELHUMID'].values)
                rain_sig.extend(group['RAINFALL'].values)
            else:
                lats_nonsig.extend(group['LAT'].values)
                lons_nonsig.extend(group['LONG'].values)
                alt_nonsig.extend(group['ALTITUDE'].values)
                time_nonsig.extend(time_rel.values)  
                temp_nonsig.extend(group['AIR_TEMP'].tolist())
                RH_nonsig.extend(group['RELHUMID'].values)
                rain_nonsig.extend(group['RAINFALL'].values)

############################################
##--Map: Convex Hull around trajectories--##
############################################
##--INPUTS--##

##--The alpha value controls the hull fit--##
alpha = 0.5

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
            ax_map_sig.fill(x, y, facecolor='None', edgecolor='#9D0759', ls='--', lw=2,
                            transform=ccrs.PlateCarree(), zorder=5)
    
        elif hull_sig_bins.geom_type == "MultiPolygon":
            for poly in hull_sig_bins.geoms:
                if not poly.is_empty:
                    x, y = poly.exterior.xy
                    ax_map_sig.fill(x, y, facecolor='None', edgecolor='#9D0759', ls='--', lw=2,
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
    ax_map_nonsig.fill(x, y, facecolor='None', edgecolor='#9D0759', ls='--', lw=2,
                       transform=ccrs.PlateCarree(), zorder=7)

elif hull_nonsig_bins.geom_type == "MultiPolygon":
    for poly in hull_nonsig_bins.geoms:
        if not poly.is_empty:
            x, y = poly.exterior.xy
            ax_map_nonsig.fill(x, y, facecolor='None', edgecolor='#9D0759', ls='--',
                       linewidth=2, transform=ccrs.PlateCarree(), zorder=7)
else:
    print("Alpha shape is not a polygon:", hull_nonsig_bins.geom_type)


########################
##--Histogram set up--##
########################

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

##--Count the number of datapoints in each bin--##
sig_count, _, _ = np.histogram2d(time_sig, alt_sig_arr, bins=(time_bins_rel, alt_bins))
nonsig_count, _, _ = np.histogram2d(time_nonsig, alt_nonsig_arr, bins=(time_bins_rel, alt_bins))



##--Plot formatting--##
for ax in [ax_temp_sig, ax_temp_nonsig,
           ax_RH_sig, ax_RH_nonsig,
           ax_rain_sig, ax_rain_nonsig]:
    ax.set_xlim(time_min, time_max)  # 0 on right
    ax.tick_params(axis='both', labelsize=16)

###################
##--Temperature--##
###################

##--Convert temperature lists to arrays--##
temp_sig_arr = np.array(temp_sig)
temp_nonsig_arr = np.array(temp_nonsig)

##--Sum all temperature values in each histogram bin--##
temp_sig_sum, _, _ = np.histogram2d(time_sig, alt_sig_arr, 
                        bins=(time_bins_rel, alt_bins), weights=temp_sig_arr)
temp_nonsig_sum, _, _ = np.histogram2d(time_nonsig, alt_nonsig,
                        bins=(time_bins_rel, alt_bins), weights=temp_nonsig_arr)

##--Compute the average RH per bins with counts--##
temp_sig_mean = np.divide(temp_sig_sum, sig_count, 
                out=np.full_like(temp_sig_sum, np.nan), where=sig_count > 0)
temp_nonsig_mean = np.divide(temp_nonsig_sum, nonsig_count, 
                out=np.full_like(temp_nonsig_sum, np.nan), where=nonsig_count > 0)

##--Plot alitude vs temp--##
mesh_sig = ax_temp_sig.pcolormesh(time_bins_rel, alt_bins, temp_sig_mean.T,
                       cmap='magma', alpha=0.75, edgecolors='none', shading='auto',
                       vmin=220, vmax=300)
mesh_nonsig = ax_temp_nonsig.pcolormesh(time_bins_rel, alt_bins, temp_nonsig_mean.T,
                          cmap='magma', alpha=0.75, edgecolors='none', shading='auto',
                          vmin=220, vmax=300)

##--Remove tick labels - share with bottom plots--##
for ax in [ax_temp_sig, ax_temp_nonsig]:
    ax.tick_params(axis='x', labelbottom=False)
    ax.tick_params(axis='both', labelsize=16) 

##--Add one colorbar--##
cbar3 = plt.colorbar(mesh_sig, cax=cax_temp,
                    orientation='vertical', fraction=0.065, pad=0.0)
cbar3.set_label('Temperature (K)', size=12)
cbar3.ax.tick_params(labelsize=14)

##########
##--RH--##
##########

##--Convert RH lists to arrays--##
RH_sig_arr = np.array(RH_sig)
RH_nonsig_arr = np.array(RH_nonsig)

##--Sum all RH values in each histogram bin--##
RH_sig_sum, _, _ = np.histogram2d(time_sig, alt_sig_arr, 
                        bins=(time_bins_rel, alt_bins), weights=RH_sig_arr)
RH_nonsig_sum, _, _ = np.histogram2d(time_nonsig, alt_nonsig,
                        bins=(time_bins_rel, alt_bins), weights=RH_nonsig_arr)

##--Compute the average RH per bins with counts--##
RH_sig_mean = np.divide(RH_sig_sum, sig_count, 
                out=np.full_like(RH_sig_sum, np.nan), where=sig_count > 0)
RH_nonsig_mean = np.divide(RH_nonsig_sum, nonsig_count, 
                out=np.full_like(RH_nonsig_sum, np.nan), where=nonsig_count > 0)

##--Plot RH--##
mesh_RH_sig = ax_RH_sig.pcolormesh(time_bins_rel, alt_bins, RH_sig_mean.T,
                                   cmap='viridis', alpha=0.75, shading='auto',
                                   vmin=0, vmax=105)
mesh_RH_nonsig = ax_RH_nonsig.pcolormesh(time_bins_rel, alt_bins, RH_nonsig_mean.T,
                                         cmap='viridis', alpha=0.75, shading='auto',
                                         vmin=0, vmax=105)

##--Remove tick labels - share with bottom plots--##
for ax in [ax_RH_sig, ax_RH_nonsig]:
    ax.tick_params(axis='x', labelbottom=False)
    ax.tick_params(axis='both', labelsize=16) 

##--Add one colorbar--##
cbar4 = plt.colorbar(mesh_RH_sig, cax=cax_RH,
                     orientation='vertical', fraction=0.065, pad=0.0)
cbar4.set_label('Relative Humidity (%)', size=12)
cbar4.ax.tick_params(labelsize=14)

################
##--Rainfall--##
################

##--Convert rainfall lists to arrays--##
rain_sig_arr = np.array(rain_sig)
rain_nonsig_arr = np.array(rain_nonsig)

##--Sum of rainfall (in mm/hr) per each bin--##
rain_sig_sum, _, _ = np.histogram2d(time_sig, alt_sig_arr, 
                        bins=(time_bins_rel, alt_bins), weights=rain_sig_arr)
rain_nonsig_sum, _, _ = np.histogram2d(time_nonsig, alt_nonsig,
                        bins=(time_bins_rel, alt_bins), weights=rain_nonsig_arr)

##--Average rainfall per bin for values greater than 0--##
rain_sig_mean = np.divide(rain_sig_sum, sig_count, 
                out=np.full_like(rain_sig_sum, np.nan), where=sig_count > 0)
rain_nonsig_mean = np.divide(rain_nonsig_sum, nonsig_count, 
                out=np.full_like(rain_nonsig_sum, np.nan), where=nonsig_count > 0)

##--Rainfall curtain plots--##
mesh_rain_sig = ax_rain_sig.pcolormesh(time_bins_rel, alt_bins, rain_sig_mean.T,
                                   cmap='Blues', alpha=0.75, shading='auto',
                                   vmin=0, vmax=2)
mesh_rain_nonsig = ax_rain_nonsig.pcolormesh(time_bins_rel, alt_bins, rain_nonsig_mean.T,
                                         cmap='Blues', alpha=0.75, shading='auto', 
                                         vmin=0, vmax=2)

##--Add x-axis labels--##
for ax in [ax_rain_sig, ax_rain_nonsig]: 
    ax.set_xlabel("Days before measurement", fontsize=18)

##############################
##--Finishing plot touches--##
##############################

##--Set up function to grey out empty plots--##
def grey_plots(ax):
    
    ##--Add a grey rectangle covering the whole axes--##
    ax.add_patch(mpatches.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                 facecolor='lightgrey', alpha=0.6, zorder=7))

##--Suggestion from GPT 5 model for removing labels when there is no data--##
for ax, has_data in [(ax_temp_sig, len(time_sig) > 0 and len(alt_sig) > 0),
                     (ax_temp_nonsig, True),   # nonsig always has data
                     (ax_RH_sig, len(time_sig) > 0 and len(alt_sig) > 0),
                     (ax_RH_nonsig, True),
                     (ax_rain_sig, len(time_sig) > 0 and len(alt_sig) > 0),
                     (ax_rain_nonsig, True)]:

    if has_data:
        ##--X-axis is now relative days (numeric), so just rotate labels--##
        ax.set_xlim(time_min, time_max)  # keep 0 on right
        ax.tick_params(axis='both', labelsize=16)
        ax.set_yticks(np.arange(0, 10000, 2000))
    else:
        ##--Grey out or blank axes when no data is present--##
        ax.set_xticks([])
        ax.set_yticks(np.arange(0, 10000, 2000))
        ax.tick_params(axis='both', labelsize=16)
        ax.set_xlabel("")
        ax.set_ylabel("")

##--Grey out any completely empty sig plots--##
if len(time_sig) == 0 or len(alt_sig) == 0:
    grey_plots(ax_map_sig)
    grey_plots(ax_temp_sig)
    grey_plots(ax_RH_sig)
    grey_plots(ax_rain_sig)

##--Add one colorbar--##
cbar5 = plt.colorbar(mesh_rain_sig, cax=cax_rain,
                     orientation='vertical', fraction=0.065, pad=0.0)
cbar5.set_label('Rainfall (mm/hr)', size=12)
cbar5.ax.tick_params(labelsize=14)

##--Add text labels to each set of plots--##
ax_map_sig.text(0.50, 0.91, 'Nanophytoplankton', horizontalalignment='center', 
         verticalalignment='center', transform=ax_map_sig.transAxes, fontsize=18,
         bbox=dict(boxstyle="round, pad=0.5", fc="white", ec='none', lw=1, alpha=0.75,
                   zorder=8))
ax_temp_sig.text(0.78, 0.9, 'Temperature', horizontalalignment='center', 
         verticalalignment='center', transform=ax_temp_sig.transAxes, fontsize=18,
         zorder=8)
ax_RH_sig.text(0.70, 0.9, 'Relative Humidity', horizontalalignment='center', 
         verticalalignment='center', transform=ax_RH_sig.transAxes, fontsize=18,
         zorder=8)
ax_rain_sig.text(0.87, 0.9, 'Rainfall', horizontalalignment='center', 
         verticalalignment='center', transform=ax_rain_sig.transAxes, fontsize=18,
         zorder=8)

plt.show()

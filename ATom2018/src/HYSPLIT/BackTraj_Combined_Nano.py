# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 21:39:28 2026

@author: repooley
"""

from pathlib import Path
import os
import glob
import hashlib
import numpy as np
import numpy.ma as ma
import pandas as pd
import xarray as xr # to read in .nc files
import cmcrameri 
import rioxarray as rio # use to downsample geospatial data
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LogNorm
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
from datetime import datetime
from pyhdf.SD import SD, SDC
from scipy.stats import gaussian_kde

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

##--Choose which flights to analyze here!--##
##--FLIGHT1 HAS NO USHAS FILE--##

flights_to_analyze = ["Flight2", "Flight10", "Flight11", "Flight12"]

high_lat = ["Flight10", "Flight11", "Flight12"]

low_lat = ["Flight2"]

##--Pull topography from NETCARE--##
##--0.5 deg topography data from TEMIS 2010: www.temis.nl/data/gmted2010/index.php--##
topography = xr.open_dataset(r"C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data\raw\GMTED2010_15n030_0125deg.nc")

##--Pull BB emissions--##
bb_emissions = xr.open_dataset(r"C:\Users\repooley\REP_PhD\Arctic_NPF\GFED5.1_monthly_2018.nc")

###################################################
##--Pull Chlorophyll, Mixed Depth, and CO files--##
###################################################

##--Create directory based on selected flight--##
def find_flights(directory, flight):
    flight_dir = os.path.join(directory, flight)
    return flight_dir

##--Define function that creates datasets from filenames--##
def find_files(directory, flight, folder, partial_name):

    flight_dir = os.path.join(directory, flight)
    search_pattern = os.path.join(flight_dir, folder, partial_name)
    return sorted(glob.glob(search_pattern))


##--Suggestion from GPT5: compute file hash--##
def file_md5(path, block_size=65536):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()

##--de-duplicate using file hash--##
def unique_by_hash(paths):
    seen = set()
    unique = []
    for p in paths:
        h = file_md5(p)
        if h not in seen:
            seen.add(h)
            unique.append(p)
        else:
            print(f"Skipping duplicate (hash match): {p}")
    return unique

##--List directories to concatenate later--##
mixdepth_dirs = []
chlor_dirs = []

for flight in flights_to_analyze: 
    
    ##--Daily chorophyll data from Aqua MODIS satellite accessed via NASA EarthData search--##
    chlor_dir = find_files(directory, flight, "Chlorophyll", "*CHL*chlor_a*.nc")
    
    ##--Only keep unique chlorophyll files--##
    chlor_dir = unique_by_hash(chlor_dir)
    
    ##--Pull datasets individually--##
    for file in chlor_dir: 
        ds = xr.open_dataset(file)
        chlor_dirs.append(ds)
        
    ##--USE 1998 DATA FOR NOW, use updated later--##
    #mixdepth_dir = r"C:\Users\repooley\REP_PhD\Arctic_NPF\FIREACE1998\data\raw\OCEAN_MIXED_LAYER_DEPTH_mon_mean_1998-04_ECCO_V4r4_latlon_0p50deg.nc"
    #mixdepth_dir = unique_by_hash(mixdepth_dir)
    
    #for file in mixdepth_dir:
    ds = xr.open_dataset(r"C:\Users\repooley\REP_PhD\Arctic_NPF\FIREACE1998\data\raw\OCEAN_MIXED_LAYER_DEPTH_mon_mean_1998-04_ECCO_V4r4_latlon_0p50deg.nc")
    mixdepth_dirs.append(ds)
    
##--Average chlorophyll and mixdepth datasets--##
if chlor_dirs:
    chlor_combined = xr.concat(chlor_dirs, dim="file")
    chlorophyll = chlor_combined.mean(dim="file")
else:
    print("No chlorophyll data found.")
    chlorophyll = None

if mixdepth_dirs:
    mixdepth_combined = xr.concat(mixdepth_dirs, dim="file")
    mixing_combined = mixdepth_combined.mean(dim="file")
else:
    print("No mixing depth data found.")
    mixing_combined = None

##--Pull topography data--##
top_lon = topography['longitude']
top_lat = topography['latitude']
elevation = topography['elevation']

CO_paths = []

##--Get directories for CO data--##
for flight in flights_to_analyze: 
    
    CO_path = find_files(directory, flight, "CO", "*MERRA2*.nc4")
    ##--Make sure to add CO to the list as a path--##
    CO_paths.extend(CO_path)


##--Collect sized-down CO data here--##
CO_fields = []

##--Average CO datasets--##
for path in CO_paths:
    
    ##--Open the dataset--##
    ds = xr.open_dataset(path)

    ##--Pull CO concentration and pressure thickness vars--##
    ##--Pull CO concentration and pressure thickness vars--##
    CO = ds["CO"]
    DP = ds["DELP"] / 100  # pressure thickness, converted to hPa from Pa

    CO_weighted = (CO * DP).sum(dim="lev") / DP.sum(dim="lev")

    ##--Take the mean with time to shrink to two dims--##
    ##--Convert to ppb while here--##
    CO_2d = CO_weighted.mean(dim="time") *1e9

    CO_fields.append(CO_2d)

CO_combined = xr.concat(CO_fields, dim="flight")
CO_mean = CO_combined.mean(dim="flight")

##--Filter out high skewing outliers from CO data--##
CO_thresh = CO_mean.quantile(0.99)
CO_mean = CO_mean.where(CO_mean <= CO_thresh)

#############################################
##--Calculate Nanophytoplankton abundance--##
#############################################

##--See onlinelibrary.wiley.com/doi/abs/10.1029/2005JC003207 for method--##

##--Pull chlor-a and mixing depth variables--##
chlor_a = chlorophyll['chlor_a']

#mixing_depth = mixing_combined['MXLDEPTH']

##--Take the mean of the time dimension in mixing_combined--##
mixing_mean = mixing_combined.mean(dim="time")

##--Pull mixing depth--##
mixing_depth = mixing_mean["MXLDEPTH"]

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
    
        ##--Compare chlorophyll concentrations to mix level--##
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

##--Pull concentration variable for BB emission data--##
bb_conc = bb_emissions["C"]

##--Use May fire data--##
bb_time = bb_conc.sel(time="2018-05-01")

##--Average the data across the time dimension--##
bb_mean = bb_time.mean(dim="time")

##--Separate the axes from the figure object to apply different projections--##
fig = plt.figure(figsize=(6, 10), constrained_layout=True)

##--Use gridspec to access figure layout--##
gs = gridspec.GridSpec(nrows=4, ncols=1, height_ratios=[2, 2, 1, 1], 
                       figure=fig)
fig.set_constrained_layout_pads(hspace=0.01)

##--Create a map with polar stereographic projection in first two subplots--##
##--First column: NPF is significant--##
ax_binnedmap_sig = fig.add_subplot(gs[0, 0], 
                projection=ccrs.NorthPolarStereo(central_longitude=-90))
ax_map_sig = fig.add_subplot(gs[1, 0], 
                projection=ccrs.NorthPolarStereo(central_longitude=-90))
ax_temp_sig = fig.add_subplot(gs[2, 0])
ax_rain_sig = fig.add_subplot(gs[3, 0])

##--Get topography cmap and set low elevation to no color--##
colors = ['#355E3B', '#FFFDD0'] # hunter green and cream
cmap = LinearSegmentedColormap.from_list('TwoColorGradient', colors, N=256)
cmap.set_under('none') # transparent
##--Normalize the cmap, transparent under 1 m elev--##
norm = plt.Normalize(vmin=1, vmax=6000)

##--Get chlorophyll cmap (zeros are ocean colored)--##
cmap2 = cmcrameri.cm.navia
cmap2.set_under('none')

##--Get fire emissions cmap--##
cmap3 = cmcrameri.cm.lajolla

##--Specify the map projection as NorthPolarStereo--##
proj = ccrs.NorthPolarStereo(central_longitude=-90)

##--Basic map formatting--##
for ax_map in [ax_map_sig]:
    
    ##--Help from google search for trimming map bounds to a rectangle--##
    ##--Specify the lat and lon limits--##
    lon_lims = [-180, 90]
    lat_lims = [30, 90]
    
    ##--Set map bounds--##
    ax_map.set_extent([lon_lims[0], lon_lims[1], lat_lims[0], lat_lims[1]], 
                      crs=ccrs.PlateCarree())

    rect = mpath.Path([
    [0.0, 0.1],   # bottom-left
    [1.0, 0.1],   # bottom-right
    [1.0, 0.97],   # top-right
    [0.0, 0.97],   # top-left
    [0.0, 0.15]
    ])
    
    ax_map.set_boundary(rect, transform=ax_map.transAxes)
    
    ax_map.set_aspect('auto')
    
    ##--Add ocean layer to maps--##
    ax_map.add_feature(cfeature.OCEAN, fc='#F0FFFF', ec='k', lw=0.2, zorder=1)
    ax_map.add_feature(cfeature.COASTLINE, ec='k', lw=0.3, zorder=2)

    ##--Add topographic data to maps--##
    elev_map = ax_map.pcolormesh(top_lon, top_lat, elevation, 
        transform=ccrs.PlateCarree(), # match transform to projection used
        cmap=cmap, norm=norm, shading='auto', zorder=1) 
    
    emiss_map = ax_map.pcolormesh(
            bb_mean["lon"],
            bb_mean["lat"],
            bb_mean,
            transform=ccrs.PlateCarree(),
            cmap=cmap3,
            norm=LogNorm(vmin=1E5, vmax=1e12), 
            shading="auto",
            zorder=1)

    ##--Add nanophytoplankton data to maps--##
    chlor_map = ax_map.pcolormesh(chlor_lon, chlor_lat, nano_abundance,
          transform=ccrs.PlateCarree(), cmap=cmap2, alpha=0.8, 
          norm=LogNorm(vmin=0.001, vmax=100), shading='auto', zorder=1)
    
##--Format the distribution maps--##
for ax_binnedmap in [ax_binnedmap_sig]:
    
    ##--Help from google search for trimming map bounds to a rectangle--##
    ##--Specify the lat and lon limits--##
    lon_lims = [-180, 90]
    lat_lims = [30, 90]
    
    ##--Set map bounds--##
    ax_binnedmap.set_extent([lon_lims[0], lon_lims[1], lat_lims[0], lat_lims[1]], 
                      crs=ccrs.PlateCarree())

    rect = mpath.Path([
    [0.0, 0.1],   # bottom-left
    [1.0, 0.1],   # bottom-right
    [1.0, 0.97],   # top-right
    [0.0, 0.97],   # top-left
    [0.0, 0.15]
    ])
    
    ax_binnedmap.set_boundary(rect, transform=ax_binnedmap.transAxes)
    
    ax_binnedmap.set_aspect('auto')
    
    ##--Add CO data to maps--##
    CO_map = ax_binnedmap.pcolormesh(
        CO_mean["lon"],
        CO_mean["lat"],
        CO_mean,
        transform=ccrs.PlateCarree(),
        cmap=cmcrameri.cm.batlow, 
        alpha=1,
        vmin=30, 
        vmax=140,
        shading='auto')
    
    ##--Add ocean layer to maps--##
    #ax_binnedmap.add_feature(cfeature.OCEAN, fc='#F0FFFF', ec='k', lw=0.2, zorder=1)
    ##--Keep coastlines--##
    ax_binnedmap.add_feature(cfeature.COASTLINE, ec='k', lw=0.75, zorder=2)
    
    ##--Add gridlines--##
    ax_binnedmap.gridlines(draw_labels=False, lw=0.75)
    
    '''
    ##--Add topographic data to maps--##
    elev_map = ax_binnedmap.pcolormesh(top_lon, top_lat, elevation, 
        transform=ccrs.PlateCarree(), # match transform to projection used
        cmap=cmap, norm=norm, shading='auto', zorder=0) 
    '''
##--Set labels--##
ax_binnedmap_sig.set_title("2018", fontsize=24)

##--Map colorbars--##
##--Make separate axes for all colorbars--##
##--For space reasons, put elev cbar to left of top plot--##
cax1 = fig.add_axes([0.12, 0.60, 0.02, 0.18])   # [left, bottom, width, height]
cb1 = fig.colorbar(elev_map, cax=cax1, orientation="vertical", shrink=0.1,
                   ticks=[0, 2000, 4000, 6000])
cb1.set_label("Elevation (m)", size=14, fontweight='normal')
cb1.ax.yaxis.set_label_position('left')
cb1.ax.tick_params(labelsize=14, left=True, right=False, labelleft=True, 
                   labelright=False)

##--Fire emissions to left of bottom plot--##
cax2 = fig.add_axes([0.12, 0.40, 0.02, 0.18])
cb2 = fig.colorbar(emiss_map, cax=cax2, orientation='vertical', shrink=0.1,
                   ticks=[1e6, 1e8, 1e10, 1e12])
cb2.ax.yaxis.set_label_position('left')
cb2.set_label("Fire Emissions (g C/m²)", size=14, fontweight='normal')
cb2.ax.tick_params(which='both', labelsize=14, left=True, right=False, labelleft=True, 
                   labelright=False)

##--Nanophytoplankton to right of bottom plot--##
cax3 = fig.add_axes([0.82, 0.44, 0.02, 0.2])   # same x position, lower bottom
cb3 = fig.colorbar(chlor_map, cax=cax3, orientation="vertical", shrink=0.1)
cb3.set_label(r"Nano$_{\rm surf}$ (mg/m$^3$)", size=14, fontweight='normal')
cb3.ax.tick_params(labelsize=14)

##--CO to left of top plot--##
cax4 = fig.add_axes([0.12, 0.80, 0.02, 0.18])
cbar4 = fig.colorbar(CO_map, cax=cax4, orientation='vertical', shrink=0.1,
                     ticks=[40, 60, 80, 100, 120, 140])
cbar4.ax.tick_params(labelsize=14)
cbar4.ax.yaxis.set_label_position('left')
cbar4.set_label('CO (ppbv)', size=14, fontweight='normal')
cbar4.ax.tick_params(which='both', labelsize=14, left=True, right=False, labelleft=True, 
                   labelright=False)

##--Set axes limits for curtain plots below maps--##
ax_temp_sig.set_ylim(-250, 10000)
ax_rain_sig.set_ylim(-250, 10000)

##########################
##--Group trajectories--##
##########################

##--Sort trajectory outputs into signficant or non-significant NPF lists--##
lats_sig = []
lons_sig = []
magl_sig = []
masl_sig = []
time_sig = []
temp_sig = []
RH_sig = []
rain_sig = []

##--Analyze on a flight-by-flight level--##
for flight in flights_to_analyze:

    flight_directory = find_flights(HYSPLIT, flight)
    
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
        
        ##--Add second conition for top 25th percentile--##        
        ##--Determine which axis to use (NPF vs non-NPF)--##
        is_significant = pd.notna(row.nuc_significant) and row.nuc_significant > 1352
            
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
            
            ##--Expect no header for the other flights--##
            df = pd.read_csv(os.path.join(flight_directory, file),
                sep=r'\s+',
                header=None,
                skiprows=1)
            
            df.columns = columns[:df.shape[1]]
        
        ##--Basic cleaning--##
        if df.empty or 'YEAR' not in df.columns:
            print(f"Skipping bad file: {file}")
            continue
        
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
            
            ##--Suggestion from GPT5 - deal with HYSPLIT wrapping around meridian--##
            # Normalize to -180 to 180 range
            group['LONG'] = ((group['LONG'] + 180) % 360) - 180
            
            lon = group['LONG'].values
            lat = group['LAT'].values
            magl = group['ALTITUDE'].values
            temps = group['AIR_TEMP'].tolist() 
            RHs = group['RELHUMID'].values
            rain = group['RAINFALL'].values
            
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
                    masl = np.insert(masl, j + 1, np.nan)
                    time_rel = np.insert(time_rel, j + 1, np.nan)
                    temps = np.insert(temps, j + 1, np.nan)
                    RHs = np.insert(RHs, j + 1, np.nan)
                    rain = np.insert(rain, j + 1, np.nan)
    
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
                color = "#4B0082"
                linewidth = 1.25
                alpha = 1
                zorder = 5
            else:
                color = 'none'
            
            if traj_num ==1 and is_significant and flight in high_lat:
                

                ax_map.plot(lon[:index_end],
                            lat[:index_end],
                            transform=ccrs.PlateCarree(),
                            c="k", lw=1, alpha=alpha, zorder=zorder)
            
                ax_temp_sig.plot(time_rel[:index_end],
                    group['ALTITUDE'].iloc[:index_end],
                    c="k", lw=1.5, alpha=0.4, zorder=zorder)
                
                ax_rain_sig.plot(time_rel[:index_end],
                    group['ALTITUDE'].iloc[:index_end],
                    c="k", lw=1.5, alpha=0.4, zorder=zorder)
                    
            elif traj_num==1 and is_significant and flight in low_lat: 
                ax_map.plot(lon[:index_end],
                            lat[:index_end],
                            transform=ccrs.PlateCarree(),
                            c='midnightblue', lw=1, alpha=1, zorder=zorder)
            
                ax_temp_sig.plot(time_rel[:index_end],
                    group['ALTITUDE'].iloc[:index_end],
                    c='midnightblue', lw=1.5, alpha=0.6, zorder=zorder)
                
                ax_rain_sig.plot(time_rel[:index_end],
                    group['ALTITUDE'].iloc[:index_end],
                    c='midnightblue', lw=1.5, alpha=0.6, zorder=zorder)
            
            if is_significant:     
                lats_sig.extend(lat)
                lons_sig.extend(lon)
                magl_sig.extend(magl)
                time_sig.extend(time_rel) 
                temp_sig.extend(temps)
                RH_sig.extend(RHs)
                rain_sig.extend(rain)
                masl_sig.extend(masl)

##########################################
##--Map: lat/lon binned traj frequency--##
##########################################

##--White/black cbar for binned data--##
colors2 = ['black', 'white'] 
binned_cmap = LinearSegmentedColormap.from_list('TwoColorGradient', colors2, N=256)

##--INPUTS--##

##--The alpha value controls the hull fit--##
alpha = 0.2

##--Set min/max latitudes for binning in degrees--##
lat_min = 30    
lat_max = 90  

##--Set number of lon/lat bins--##
num_theta = 50  # longitude bins
num_r = 25       # latitude bins      

##--Longitude edges (uniform degrees)--##
lon_edges = np.linspace(-180, 180, num_theta + 1)

##--Latitude edges uniform in sin(lat) for equal area--##
sin_edges = np.linspace(np.sin(np.deg2rad(lat_min)),
                        np.sin(np.deg2rad(lat_max)),
                        num_r + 1)
lat_edges = np.rad2deg(np.arcsin(sin_edges))


##--Histograms in lon/lat space--##
H_sig, _, _ = np.histogram2d(lons_sig, lats_sig, bins=(lon_edges, lat_edges))

##--Compute bin areas on a unit sphere--##
lon_rads = np.deg2rad(lon_edges)
lat_rads = np.deg2rad(lat_edges)

dlam = np.diff(lon_rads)                 # Δλ for each longitude bin
dsinphi = np.diff(np.sin(lat_rads))      # Δ(sin φ) for each latitude bin

areas = np.outer(dlam, dsinphi)          # (num_theta, num_r) bin areas

##--Normalize histograms to counts per unit area--##
H_sig_density = H_sig / areas

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
    bin_colors = ax_binnedmap_sig.pcolormesh(lon_grid, lat_grid, H_sig_masked, cmap=binned_cmap, 
       norm=LogNorm(vmin=0.001, vmax=2),
        alpha=0.65, edgecolors='none', transform=ccrs.PlateCarree(), zorder=4)

else: 
    print("No data passes significance test")

##--Add colorbar with its own axis to top plot--##
cbar = plt.colorbar(bin_colors, ax=ax_binnedmap_sig, orientation='vertical', shrink=0.75)
cbar.ax.tick_params(labelsize=14)
cbar.set_label('% Trajectory Frequency', size=14, fontweight='normal')

########################
##--Histogram set up--##
########################

##--These bin numbers apply to ALL curtain plots--##
num_time_bins = 10
num_alt_bins = 8

##--Convert altitude lists to arrays--##
alt_sig_arr = np.array(masl_sig)

##--And time--##
time_sig = np.asarray(time_sig)

##--Suggestion from GPT 5 model - pad mismatched arrays to align--##
def pad_to_match(a, b):
    L = max(len(a), len(b))
    a = np.pad(a, (0, L - len(a)), constant_values=np.nan)
    b = np.pad(b, (0, L - len(b)), constant_values=np.nan)
    return a, b

time_sig, alt_sig_arr = pad_to_match(time_sig, alt_sig_arr)

##--Ensure no NaN values in any array--##
def clean_valid_pairs(time_arr, val_arr):
    mask = np.isfinite(time_arr) & np.isfinite(val_arr)
    return time_arr[mask], val_arr[mask]

time_sig, alt_sig_arr = clean_valid_pairs(time_sig, alt_sig_arr)

##--Determine overall min/max relative times for binning--##
all_time_rel = time_sig
time_min = all_time_rel.min()  # earliest day (most negative)
time_max = 0                   # measurement time = 0 days

##--Create bin edges in the time dimension--##
time_bins_rel = np.linspace(time_min, time_max, num_time_bins + 1)

##--Concatenate altitudes for bin edges--##
alt_bins = np.linspace(alt_sig_arr.min(), alt_sig_arr.max(), num_alt_bins + 1)

##--Count the number of datapoints in each bin--##
sig_count, _, _ = np.histogram2d(time_sig, alt_sig_arr, bins=(time_bins_rel, alt_bins))

##--Plot formatting--##
for ax in [ax_temp_sig, ax_rain_sig]:
    ax.set_xlim(time_min, time_max)  # 0 on right
    ax.set_yticks(np.arange(0, 10000, 2000))
    ax.tick_params(axis='both', labelsize=16)
    
###################
##--Temperature--##
###################

##--Convert temperature lists to arrays--##
temp_sig_arr = np.array(temp_sig)

##--Define a new function to clean just one variable, not two--##
##--Time is already cleaned.--##
def clean_valid_points(val_arr):
    mask = np.isfinite(val_arr)
    return val_arr[mask]

##--Use function to clean--##
temp_sig_arr = clean_valid_points(temp_sig_arr)

##--Sum all temperature values in each histogram bin--##
temp_sig_sum, _, _ = np.histogram2d(time_sig, alt_sig_arr, 
                        bins=(time_bins_rel, alt_bins), weights=temp_sig_arr)

##--Compute the average RH per bins with counts--##
temp_sig_mean = np.divide(temp_sig_sum, sig_count, 
                out=np.full_like(temp_sig_sum, np.nan), where=sig_count > 0)

##--Plot alitude vs temp--##
mesh_sig = ax_temp_sig.pcolormesh(time_bins_rel, alt_bins, temp_sig_mean.T,
                       cmap='magma', alpha=0.75, edgecolors='none', shading='auto',
                       vmin=220, vmax=300)

##--Remove tick labels - share with bottom plots--##
ax_temp_sig.tick_params(axis='x', labelbottom=False)
ax_temp_sig.tick_params(axis='both', labelsize=16) 

##--Add elevation label--##
ax_temp_sig.set_ylabel("Altitude (m)", size=14, fontweight='normal')

##--Add one colorbar to righthand colorbar ax--##
cbar2 = plt.colorbar(mesh_sig, ax = ax_temp_sig,
                    orientation='vertical', fraction=0.075, pad=0.1, 
                    ticks=[220, 240, 260, 280, 300])
cbar2.set_label('Temperature (K)', size=14, labelpad=10, fontweight='normal')
cbar2.ax.tick_params(labelsize=14)

################
##--Rainfall--##
################

##--Convert rainfall lists to arrays--##
rain_sig_arr = np.array(rain_sig)

##--Use function to clean--##
rain_sig_arr = clean_valid_points(rain_sig_arr)

##--Sum of rainfall (in mm/hr) per each bin--##
rain_sig_sum, _, _ = np.histogram2d(time_sig, alt_sig_arr, 
                        bins=(time_bins_rel, alt_bins), weights=rain_sig_arr)

##--Average rainfall per bin for values greater than 0--##
rain_sig_mean = np.divide(rain_sig_sum, sig_count, 
                out=np.full_like(rain_sig_sum, np.nan), where=sig_count > 0)

##--Rainfall curtain plots--##
mesh_rain_sig = ax_rain_sig.pcolormesh(time_bins_rel, alt_bins, rain_sig_mean.T,
                                   cmap='Blues', alpha=0.75, shading='auto',
                                   vmin=0, vmax=2)

ax_rain_sig.set_xlabel("Days before measurement", fontsize=18, fontweight='normal')
##--Add elevation label--##
ax_rain_sig.set_ylabel("Altitude (m)", size=14, fontweight='normal')

##--Add one colorbar--##
cbar3 = plt.colorbar(mesh_rain_sig, ax=ax_rain_sig,
                     orientation='vertical', fraction=0.075, pad=0.1)
cbar3.set_label('Rainfall (mm/hr)', size=14, labelpad=10, fontweight='normal')
cbar3.ax.tick_params(labelsize=14)

'''
ax_temp_sig.text(0.78, 1.1, 'Temperature', horizontalalignment='center', 
         verticalalignment='center', transform=ax_temp_sig.transAxes, fontsize=18,
         zorder=10)

ax_rain_sig.text(0.87, 1.1, 'Rainfall', horizontalalignment='center', 
         verticalalignment='center', transform=ax_rain_sig.transAxes, fontsize=18,
         zorder=10)
'''
plt.show()
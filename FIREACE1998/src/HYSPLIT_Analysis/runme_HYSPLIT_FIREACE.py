# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
Created on Mon Nov  3 08:54:30 2025

@author: (raepooley)
"""
'''
Edits made by @repooley to align with NETCARE needs

WHAT THIS SCRIPT DOES
-------------------------------------------------------------------------------
This script runs the web version of NOAA HYSPLIT Trajectory Model using the
Selenium package to control Google Chrome and save input and output files.

User needs to input RF and UTC seconds past 00Z and the code will run HYSPLIT.
As is, the script will run HYSPLIT with
 (1) forward/backward trajectory,
 (2) rounding start time (hour) up/down, and
 (3) starting height as auto mid-boundary layer (MBL),
     meters above sea level (MASL; from G_ALT), and
     meters above ground level (MAGL; G_ALT)
That said, this code is hopefully modular and relatively easy to edit to fit
all needs.


INPUT/OUTPUT FILE NAMING CONVENTION
-------------------------------------------------------------------------------
RF_UTC_WARD_ROUND_LEVEL_RUNID_TYPE.txt

    RF   Research Flight (e.g., 20230804)
   UTC   UTC seconds past midnight
  WARD   Backward (B) or forward (F) trajectory
 ROUND   Round start hour up (UP) or down (DN) (edit: automatically round to nearest hour)
 LEVEL   Start height as mid-boundary layer (MBL), MASL or MAGL
 RUNID   Unique job ID assigned by HYSPLIT
  TYPE   Input, output, or tdump (unformatted model output)


RUN DIRECTORY STRUCTURE
-------------------------------------------------------------------------------
The HYSPLIT Model run directoy should be structured as follows:
- chromedriver (from https://chromedriver.chromium.org/downloads)
> scripts
    - runme.py (this file)
> input
    - input_template.dat
    - RF_SECONDSPAST0000UTC_B/F_UP/DN_MAGL/MASL/MBL_RUNID_input.txt
> output
    - output_template.dat
    - RF_SECONDSPAST0000UTC_B/F_UP/DN_MAGL/MASL/MBL_RUNID_output.txt
    > tdump
        -  RF_SECONDSPAST0000UTC_B/F_UP/DN_MAGL/MASL/MBL_RUNID_tdump.txt
'''

import numpy as np
import pandas as pd
import os
from glob import glob
from itertools import product
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
#from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import StaleElementReferenceException
import requests
import shutil

# =============================================================================
# MODIFY MODEL RUN PARAMETERS HERE
# for all unique combiations of UTC, WARD, ROUND, LEVEL
WARDs = ['B'] #['B', 'F']
# ROUNDs = ['UP', 'DN']
LEVELs = ['MASL'] #['MBL', 'MASL'] # starting level
hide = True # show or hide the window? (True/False) (False useful for debugging)
load_data = True # Load AEROMMA data or is it already loaded as df? (Slow)

if load_data:
    df = pd.read_parquet(r"C:\Users\repooley\REP_PhD\Arctic_NPF\FIREACE1998\data\raw\FIREACE.parquet")
    # file = '../../Data/AEROMMA-MMS-1Hz_All/AEROMMA-MMS-1Hz_DC8_20230801_R0.ict' # <- maybe you have to change the directory
    # df = pd.DataFrame(pnc.pncopen(file, format='ffi1001').variables)

# select run date    
RF = '19980418'   
# %%


# select run time/location in 5 minute intervals
df_flight = df[df['Flight_date'] == RF].copy()
df_flight = df_flight.sort_values('Time_start')

# count midnight crossings (Time_start decreases)
df_flight['day_offset'] = (df_flight['Time_start'].diff() < 0).cumsum()

# build corrected datetime
base_date = pd.to_datetime(RF, format='%Y%m%d')
df_flight['datetime_fixed'] = (
    base_date
    + pd.to_timedelta(df_flight['day_offset'], unit='D')
    + pd.to_timedelta(df_flight['Time_start'], unit='s')
)
start_utc = int(df_flight['Time_start'].min())
end_utc   = int(df_flight['Time_start'].max())
UTCs = list(range(start_utc, end_utc + 1, 300))  # every 5 min
    
# get seconds past UTC midnight if it's not already in data
if 'UTC_Start' not in df.columns:
    df['UTC_Start'] = df['datetime'].apply(lambda x: (x - x.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()).astype(int)

# Path for scripts
path = r'C:\Users\repooley\REP_PhD\Arctic_NPF\FIREACE1998\src'
os.chdir(path)

# Save path
output_root = r'C:\Users\repooley\REP_PhD\Arctic_NPF\FIREACE1998\data\raw\HYSPLIT'

# =============================================================================

#%% GET MODEL INPUT PARAMETERS

# this loop reads in one row at a time
for UTC, WARD in list(product(UTCs,WARDs)):
# for UTC, WARD, ROUND in list(product(UTCs,WARDs,ROUNDs)):

    LEVEL='MASL'
    if np.round(UTC/3600)==np.floor(UTC/3600): ROUND='DN'
    if np.round(UTC/3600)!=np.floor(UTC/3600): ROUND='UP'

    # Subset data
    # data = df.loc[ (df['Flight']==RF) & (df['UTC_Start']==UTC) ]
    print(RF,UTC)
    
    
    data = df_flight.loc[df_flight['Time_start'] == UTC]
    
    # Skip any NaN times
    if data.empty:
        print(f"[WARNING] No data found for RF={RF}, UTC={UTC}. Skipping.")
        continue
    
    # Check for NaNs in required columns before trying to access them
    if data[['Lat', 'Lon', 'Alt']].isnull().values.any():
        print(f"[WARNING] Missing lat/lon/alt for RF={RF}, UTC={UTC}. Skipping.")
        continue
    
    dt = data['datetime_fixed'].iloc[0]
    run_date = dt.strftime('%Y%m%d')
    run_hour = dt.strftime('%H')
    run_min  = dt.strftime('%M')
    
    print(run_date, run_hour, run_min)
    
    flight_output_dir = os.path.join(output_root, RF)
    os.makedirs(flight_output_dir, exist_ok=True)

    # (1) Type of trajectory(ies)
    num_trajector_starting_locations = 1
    type_of_trajectory = 'Ensemble'
    
    # (2) Meteorology & Starting Location(s)
    meteorology = 'reanalysis' # only compatable with HRRR, GDAS0p5, and REANALYSIS for now
    '''
    Option       Descrpition
    ------------------------------------------------------------------------------
    GDAS1        GDAS (1 degree, global, 2006-present)
    GFS0p25      GDAS (0.25 degree, global, 06/2019-present)
    GDAS0p5      GDAS (0.5 degree, global, 09/2007-06/2019)
    NAM12        NAM 12km (pressure, U.S., 05/2007-present)
    HRRR         HRRR 3km (sigma, U.S., 06/2019-present)
    HRRR.V1      HRRRV1 3km (sigma, U.S., 06/2015-07/2019)
    NAMS         NAM 12km (hybrid sigma-pressure, U.S., 03/2010-present)
    NARR         NARR 32km (N.A., 1979-2019)
    EDAS40       EDAS 40km (U.S., 2004-)
    EDAS         EDAS 80km (U.S., 1997-2004)
    NGM          NGM (N.A., 1991-1997)
    reanalysis   REANALYSIS (global, 1984-present)
    WRF27km      WRF 27km (U.S., 1980-present)
    '''
    
    coordinate_type = 'Decimal Degrees'
    
    # some data extends past midnight
    # access the row in data
    row = data.iloc[0]

    # The datetime variable has the correct date and time
    year  = row['datetime'].year
    month = row['datetime'].month
    day   = row['datetime'].day
    hour  = row['datetime'].hour
    

    latitude = data['Lat'].values[0]   # deg N
    longitude = data['Lon'].values[0] # deg E
    
    # (4) Model Run Details
    if WARD=='F': trajectory_direction = 'Forward'
    if WARD=='B': trajectory_direction = 'Backward'
    vertical_motion = 'Model vertical velocity'
    #year = data['datetime'].dt.year.values[0]
    #month = data['datetime'].dt.month.values[0]
    #day = data['datetime'].dt.day.values[0]
    #hour = data['datetime'].dt.hour.values[0]
    total_run_time_hr = 120 # 5 days
    new_trajectory_freq_hr = 0
    max_num_trajectories = 24
    latitude # should be same from before
    longitude # should be same from before
    if LEVEL=='MBL': auto_midboundary_layer_height = True
    if LEVEL!='MBL': auto_midboundary_layer_height = False
    altitude = data['Alt'].values[0]
    height_units = 'meters AMSL' # (change if using meters AGL)
    
    # (4 cont.) Display Options
    GIS_output = 'None'
    dpi = 300 # Options: 72, 96, 120, 300 
    zoom_factor = 70 # Options: 0-100
    projection = 'Polar' # Options: Default, Polar, Lambert, Mercator
    vertical_units = 'Meters AGL' # Options: Pressure, Meters AGL, Theta
    label_interval = '1 hour' # Options: No labels, 1 hour, 6 hours, 12 hours, 24 hours
    color_trajectories = False # T/F
    color_by_source = True  # T/F
    source_symbol = True # T/F
    distance_overlay = False # T/F
    bounty_borders = False  # T/F
    postscript_file = False # T/F
    pdf_file = False # T/F
    plot_met_field = False  # T/F
    met_data_terrain_m = True # T/F
    met_data_potentialT_K = True # T/F
    met_data_ambientT_K = True # T/F
    met_data_rain_mm_hr = True # T/F
    met_data_mixed_layer_depth_m = True # T/F
    met_data_RH = True # T/F
    met_data_dn_solar_flux_w_m2 = True # T/F
    
    # RUN MODEL
    print('============================')
    print('      INPUT PARAMETERS      ')
    print('----------------------------')
    
    print(f'             RF: {RF}')
    print(f'            UTC: {UTC}')
    if WARD=='F': print('     TRAJECTORY: FORWARD')
    if WARD=='B': print('     TRAJECTORY: BACKWARD')
    print(f' ROUND START HR: {ROUND}')
    print(f'         HEIGHT: {LEVEL}')
    print('============================')
    print('')
    print('RUNNING HYSPLIT...              ', end='\r')

    # Check if run already exists
    fdir = '../output'
    fids = glob(os.path.join(fdir, '*.txt'))
    UTC_str = f"{UTC:05d}"   # pad to 5 digits
    run_id = f"{RF}_{UTC_str}_{WARD}_{ROUND}_{LEVEL}"
    
    run_exists = any(run_id in os.path.basename(f) for f in fids)      
    
    if run_exists:
        print('RUN EXISTS. MOVING ON...', end='\r')
        print('')
    else:
    # if True:
        '''
        Much help from this tutorial:
        https://towardsdatascience.com/controlling-the-web-with-python-6fceb22c5f08
        and Google/Stackoverflow where my needs differ
        '''
        # Open E-AIM Model IV Batch Mode Website
        '''
        Note: must download chromedriver (assuming you're using Google Chrome) from
        https://chromedriver.chromium.org/downloads and put it in HYSPLIT directory
        '''
        # Use Chrome to access web

        opt = Options()
        if hide: 
            opt.headless = True
        
        # Use webdriver_manager to manage chromedriver
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=opt)
        
        # Go to website
        driver.get('https://www.ready.noaa.gov/hypub-bin/trajtype.pl?runtype=archive')
        
        # (1) Type of Trajectory(ies)
        # Number of Trajectory Starting Locations
        driver.find_element(by=By.CSS_SELECTOR, value='input[type="RADIO"][name="nsrc"][value="%i"]'%num_trajector_starting_locations).click()
        # Type of Trajectory
        if type_of_trajectory=='Normal': driver.find_element(by=By.CSS_SELECTOR, value='input[type="RADIO"][name="trjtype"][value="1"]').click()
        if type_of_trajectory=='Matrix': driver.find_element(by=By.CSS_SELECTOR, value='input[type="RADIO"][name="trjtype"][value="2"]').click()
        if type_of_trajectory=='Ensemble': driver.find_element(by=By.CSS_SELECTOR, value='input[type="RADIO"][name="trjtype"][value="3"]').click()
        if type_of_trajectory=='Frequency': driver.find_element(by=By.CSS_SELECTOR, value='input[type="RADIO"][name="trjtype"][value="4"]').click()
        # Next>>
        driver.find_element(by=By.CSS_SELECTOR, value='input[type="submit"][value="Next>>"]').click()
        
        # (2) Meteorology & Starting Location(s)
        # Meteorology
        max_wait_time = 60 # seconds
        link_present = EC.presence_of_element_located((By.CSS_SELECTOR, 'option[value="%s"]'%meteorology))
        WebDriverWait(driver, max_wait_time).until(link_present)
        driver.find_element(by=By.CSS_SELECTOR, value='option[value="%s"]'%meteorology).click()
        
        # Source location
        if coordinate_type=='Decimal Degrees':
            driver.find_element(by=By.CSS_SELECTOR, value='#LatId').clear()
            driver.find_element(by=By.CSS_SELECTOR, value='#LatId').send_keys(np.abs(latitude))
            Select(driver.find_element(By.ID, 'nsId')).select_by_visible_text('N' if latitude>0 else 'S')
            driver.find_element(by=By.CSS_SELECTOR, value='#LonId').clear()
            driver.find_element(by=By.CSS_SELECTOR, value='#LonId').send_keys(np.abs(longitude))
            Select(driver.find_element(By.ID, 'ewId')).select_by_visible_text('E' if longitude>0 else 'W')
        
        if coordinate_type=='DDD/MM/SS':
            print('Coming soon...')
        if coordinate_type=='City':
            print('Coming soon...')
        if coordinate_type=='Airport or WMO ID':
            print('Coming soon...')
        # Next>>
        max_wait_time = 10 # seconds
        link_present = EC.presence_of_element_located((By.CSS_SELECTOR, 'input[type="button"][value="Next>>"]'))
        WebDriverWait(driver, max_wait_time).until(link_present)
        driver.find_element(by=By.CSS_SELECTOR, value='input[type="button"][value="Next>>"]').click()
        
        # (3) Meteorology File
        # Choose an archived meteorological file
        if meteorology=='HRRR':
            YYYY = '%i'%year
            MM = '%02i'%month
            DD = '%02i'%day
            
            if ROUND=='DN': hr = hour
            if ROUND=='UP': hr = hour+1
            
            if  0<=hr<= 5: hr_range = '00-05'
            if  6<=hr<=11: hr_range = '06-11'
            if 12<=hr<=17: hr_range = '12-17'
            if 18<=hr<=23: hr_range = '18-23'
            
            max_wait_time = 60 # seconds
            link_present = EC.presence_of_element_located((By.TAG_NAME, 'select'))
            WebDriverWait(driver, max_wait_time).until(link_present)
            Select(driver.find_element(By.TAG_NAME, 'select')).select_by_visible_text('%s%s%s_%s_hrrr'%(YYYY,MM,DD,hr_range))
            
        if meteorology=='GDAS0p5':
            
            # Some flights extend past midnight UTC. The datetime variable is correct
            dt_model = row['datetime']
            
            # Extract formatted components
            YYYY = dt_model.strftime('%Y')   # e.g. '2015'
            MM   = dt_model.strftime('%m')   # e.g. '04'
            DD   = dt_model.strftime('%d')   # e.g. '21'
            hr   = dt_model.strftime('%H')   # e.g. '01'
            
            # HYSPLIT expects hr without leading zero
            # except when the hour is 00!
            if hr != '00':
                hr = str(int(hr)) 
            
            # Wait for dropdown to exist
            max_wait_time = 60
            wait = WebDriverWait(driver, max_wait_time)
            wait.until(EC.presence_of_element_located((By.TAG_NAME, 'select')))
            
            # Re-fetch the element and select
            try:
                dropdown_element = driver.find_element(By.TAG_NAME, 'select')  # <--- REFIND it here
                Select(dropdown_element).select_by_visible_text(f"{YYYY}{MM}{DD}_gdas0p5")
            # Add exception to re-try if request went stale
            except StaleElementReferenceException:
                print("Dropdown element went stale. Re-finding...")
                dropdown_element = driver.find_element(By.TAG_NAME, 'select')
                Select(dropdown_element).select_by_visible_text(f"{YYYY}{MM}{DD}_gdas0p5")
        
        if meteorology=='reanalysis':
            
            # Some flights extend past midnight UTC. The datetime variable is correct
            dt_model = row['datetime']
            
            # Extract formatted components
            YYYY = dt_model.strftime('%Y')   # e.g. '2015'
            MM   = dt_model.strftime('%m')   # e.g. '04'
            DD   = dt_model.strftime('%d')   # e.g. '21'
            hr   = dt_model.strftime('%H')   # e.g. '01'
            
            # HYSPLIT expects hr without leading zero
            # except when the hour is 00!
            if hr != '00':
                hr = str(int(hr)) 
            
            # Wait for dropdown to exist
            max_wait_time = 60
            wait = WebDriverWait(driver, max_wait_time)
            wait.until(EC.presence_of_element_located((By.TAG_NAME, 'select')))
            
            # Re-fetch the element and select
            try:
                dropdown_element = driver.find_element(By.TAG_NAME, 'select')  # <--- REFIND it here
                Select(dropdown_element).select_by_visible_text(f"RP{YYYY}{MM}.gbl")
            # Add exception to re-try if request went stale
            except StaleElementReferenceException:
                print("Dropdown element went stale. Re-finding...")
                dropdown_element = driver.find_element(By.TAG_NAME, 'select')
                Select(dropdown_element).select_by_visible_text(f"RP{YYYY}{MM}.gbl")
        
        else:
            print('Choose HRRR, REANALYSIS, or GDAS0p5 meteorology, this code does not support other meteorology')
        # Next>>
        link_present = EC.presence_of_element_located((By.CSS_SELECTOR, 'input[type="submit"][value="Next>>"]'))
        WebDriverWait(driver, max_wait_time).until(link_present)
        driver.find_element(by=By.CSS_SELECTOR, value='input[type="submit"][value="Next>>"]').click()

        # (4) Model Run Details (Model Parameters)
        # Trajectory Direction
        link_present = EC.presence_of_element_located((By.CSS_SELECTOR, 'input[type="RADIO"][name="direction"][value="Forward"]'))
        WebDriverWait(driver, max_wait_time).until(link_present)
        if WARD=='F': driver.find_element(by=By.CSS_SELECTOR, value='input[type="RADIO"][name="direction"][value="Forward"]').click()
        if WARD=='B': driver.find_element(by=By.CSS_SELECTOR, value='input[type="RADIO"][name="direction"][value="Backward"]').click()
        
        # Vertical Motion (skip; default Model vertical velocity)
        if vertical_motion=='Model vertical velocity':
            driver.find_element(by=By.CSS_SELECTOR, value='input[type="RADIO"][name="vertical"][value="0"]').click()
        if vertical_motion=='Isobaric':
            driver.find_element(by=By.CSS_SELECTOR, value='input[type="RADIO"][name="vertical"][value="1"]').click()
        if vertical_motion=='Isentropic':
            driver.find_element(by=By.CSS_SELECTOR, value='input[type="RADIO"][name="vertical"][value="2"]').click()
        # Start Time (UTC)
        # Year (skip; only one default option)
        Select(driver.find_element(By.NAME, 'Start year')).select_by_visible_text(YYYY[2:])
        # Month (skip; only one default option)
        Select(driver.find_element(By.NAME, 'Start month')).select_by_visible_text(MM)
        # Day (skip; only one default option)
        Select(driver.find_element(By.NAME, 'Start day')).select_by_visible_text(DD)
        # Hour
        Select(driver.find_element(By.NAME, 'Start hour')).select_by_visible_text(hr)
        
        # Total run time (hours)
        driver.find_element(By.NAME, 'duration').clear()
        driver.find_element(By.NAME, 'duration').send_keys(total_run_time_hr)
        # Start a new trajectory every (hrs)
        driver.find_element(By.NAME, 'repeatsrc').clear()
        driver.find_element(By.NAME, 'repeatsrc').send_keys(new_trajectory_freq_hr)
        # Maximum number of trajectories
        driver.find_element(By.NAME, 'ntrajs').clear()
        driver.find_element(By.NAME, 'ntrajs').send_keys(max_num_trajectories)
        
        # Start 1 latitude (deg) (skip; already filled)
        driver.find_element(By.NAME, 'Source lat').clear()
        driver.find_element(By.NAME, 'Source lat').send_keys(latitude)
        # Start 1 longitude (deg) (skip; already filled)
        driver.find_element(By.NAME, 'Source lon').clear()
        driver.find_element(By.NAME, 'Source lon').send_keys(longitude)
        
        # Auto mid-boundary layer height?
        if auto_midboundary_layer_height: 
            driver.find_element(by=By.CSS_SELECTOR, value='input[type="RADIO"][name="Midlayer height"][value="Yes"]').click()
        else:
            driver.find_element(by=By.CSS_SELECTOR, value='input[type="RADIO"][name="Midlayer height"][value="No"]').click()
        
        # Level 1 height
        driver.find_element(By.NAME, 'Source hgt1').clear()
        driver.find_element(By.NAME, 'Source hgt1').send_keys(altitude)
        
        # height units
        # height_units = 'meters AMSL' # to use G_ALT (change if using meters AGL)
        if height_units=='meters AGL': 
            driver.find_element(by=By.CSS_SELECTOR, value='input[type="RADIO"][name="Source hunit"][value="0"]').click()
        if height_units=='meters AMSL': 
            driver.find_element(by=By.CSS_SELECTOR, value='input[type="RADIO"][name="Source hunit"][value="1"]').click()
        
        
        # (4 cont.) Model Run Details (Display Options)
        # GIS output
        if GIS_output=='None':
            driver.find_element(by=By.CSS_SELECTOR, value='input[type="RADIO"][name="gis"][value="0"]').click()
        if GIS_output=='Google Earth (kmz)':
            driver.find_element(by=By.CSS_SELECTOR, value='input[type="RADIO"][name="gis"][value="3"]').click()
        if GIS_output=='GIS Shapefile':
            driver.find_element(by=By.CSS_SELECTOR, value='input[type="RADIO"][name="gis"][value="1"]').click()
        
        # Plot resolution (dpi)
        Select(driver.find_element(By.NAME, 'gsize')).select_by_visible_text(str(dpi))
        # Zoom factor
        driver.find_element(By.NAME, 'Zoom Factor').clear()
        driver.find_element(By.NAME, 'Zoom Factor').send_keys(str(zoom_factor))
        
        # Plot projection
        if projection=='Default':
            driver.find_element(by=By.CSS_SELECTOR, value='input[type="RADIO"][name="projection"][value="0"]').click()
        if projection=='Polar':
            driver.find_element(by=By.CSS_SELECTOR, value='input[type="RADIO"][name="projection"][value="1"]').click()
        if projection=='Lambert':
            driver.find_element(by=By.CSS_SELECTOR, value='input[type="RADIO"][name="projection"][value="2"]').click()
        if projection=='Mercator':
            driver.find_element(by=By.CSS_SELECTOR, value='input[type="RADIO"][name="projection"][value="3"]').click()
        
        # Vertical plot height units
        if vertical_units=='Pressure':
            driver.find_element(by=By.CSS_SELECTOR, value='input[type="RADIO"][name="Vertical Unit"][value="0"]').click()
        if vertical_units=='Meters AGL':
            driver.find_element(by=By.CSS_SELECTOR, value='input[type="RADIO"][name="Vertical Unit"][value="1"]').click()
        if vertical_units=='Theta':
            driver.find_element(by=By.CSS_SELECTOR, value='input[type="RADIO"][name="Vertical Unit"][value="2"]').click()
        
        # Label interval
        if label_interval=='No labels':
            driver.find_element(by=By.CSS_SELECTOR, value='input[type="RADIO"][name="Label Interval"][value="0"]').click()
        if label_interval=='1 hour':
            driver.find_element(by=By.CSS_SELECTOR, value='input[type="RADIO"][name="Label Interval"][value="1"]').click()
        if label_interval=='6 hours':
            driver.find_element(by=By.CSS_SELECTOR, value='input[type="RADIO"][name="Label Interval"][value="6"]').click()
        if label_interval=='12 hours':
            driver.find_element(by=By.CSS_SELECTOR, value='input[type="RADIO"][name="Label Interval"][value="12"]').click()
        if label_interval=='24 hours':
            driver.find_element(by=By.CSS_SELECTOR, value='input[type="RADIO"][name="Label Interval"][value="24"]').click()
        
        # Plot color trajectories?
        if color_trajectories:
            driver.find_element(by=By.CSS_SELECTOR, value='input[type="RADIO"][name="color"][value="Yes"]').click()
        else:
            driver.find_element(by=By.CSS_SELECTOR, value='input[type="RADIO"][name="color"][value="No"]').click()
        
        # use same colors for each source location?
        if color_by_source:
            driver.find_element(by=By.CSS_SELECTOR, value='input[type="RADIO"][name="colortype"][value="Yes"]').click()
        else:
            driver.find_element(by=By.CSS_SELECTOR, value='input[type="RADIO"][name="colortype"][value="No"]').click()
        
        # Pot source location symbol?
        if source_symbol:
            driver.find_element(by=By.CSS_SELECTOR, value='input[type="RADIO"][name="pltsrc"][value="1"]').click()
        else:
            driver.find_element(by=By.CSS_SELECTOR, value='input[type="RADIO"][name="pltsrc"][value="0"]').click()
        
        # Distance circle overlay
        if distance_overlay:
            driver.find_element(by=By.CSS_SELECTOR, value='input[type="RADIO"][name="circle"][value="0"]').click()
        else:
            driver.find_element(by=By.CSS_SELECTOR, value='input[type="RADIO"][name="circle"][value="-1"]').click()
        
        # U.S. county borders?vertical_units = 'Meters AGL' # Options: Pressure, Meters AGL, Theta
        if bounty_borders:
            driver.find_element(by=By.CSS_SELECTOR, value='input[type="RADIO"][name="county"][value="map_county_coast"]').click()
        else:
            driver.find_element(by=By.CSS_SELECTOR, value='input[type="RADIO"][name="county"][value="arlmap"]').click()
        
        # Postcript file?
        if postscript_file:
            driver.find_element(by=By.CSS_SELECTOR, value='input[type="RADIO"][name="psfile"][value="Yes"]').click()
        else:
            driver.find_element(by=By.CSS_SELECTOR, value='input[type="RADIO"][name="psfile"][value="No"]').click()
        
        # PDF file?
        if pdf_file:
            driver.find_element(by=By.CSS_SELECTOR, value='input[type="RADIO"][name="pdffile"][value="Yes"]').click()
        else:
            driver.find_element(by=By.CSS_SELECTOR, value='input[type="RADIO"][name="pdffile"][value="No"]').click()
        
        # Plot meteorological field along trajectory?
        if plot_met_field:
            driver.find_element(by=By.CSS_SELECTOR, value='input[type="RADIO"][name="mplot"][value="YES"]').click()
        else:
            driver.find_element(by=By.CSS_SELECTOR, value='input[type="RADIO"][name="mplot"][value="NO"]').click()
        
        # Dump meteorological data along trajectory
        element = driver.find_element(By.NAME, 'terr')
        if met_data_terrain_m != element.is_selected(): element.click()
        
        element = driver.find_element(By.NAME, 'tpot')
        if met_data_potentialT_K != element.is_selected(): element.click()
        
        element = driver.find_element(By.NAME, 'tamb')
        if met_data_ambientT_K != element.is_selected(): element.click()
        
        element = driver.find_element(By.NAME, 'rain')
        if met_data_rain_mm_hr != element.is_selected(): element.click()
        
        element = driver.find_element(By.NAME, 'mixd')
        if met_data_mixed_layer_depth_m != element.is_selected(): element.click()
        
        element = driver.find_element(By.NAME, 'relh')
        if met_data_RH != element.is_selected(): element.click()
        
        element = driver.find_element(By.NAME, 'dswf')
        if met_data_dn_solar_flux_w_m2 != element.is_selected(): element.click()
        
        # Request trajectory (only press once!)
        driver.find_element(by=By.CSS_SELECTOR, value='input[type="submit"][value="Request trajectory (only press once!)"]').click()

        '''
        #%% for testing
        ser = Service('../chromedriver')
        opt = webdriver.ChromeOptions()
        if hide: opt.headless = True
        driver = webdriver.Chrome(service=ser, options=opt)
        driver.get('https://www.ready.noaa.gov/hypub-bin/trajresults.pl?jobidno=161072')
        '''
        
        # wait max 2 min until model done
        max_wait_time = 300 # seconds
        link_present = EC.presence_of_element_located((By.XPATH, "//a[contains(@href,'tdump')]"))
        WebDriverWait(driver, max_wait_time).until(link_present)
        '''
        ^ in the future, it would be smarter to see what % the job is done
        running on the first page (before the first 10s refresh) and then
        estimate some np.ceil() of how many 10s chunks it will take for the
        job to finish. bc otherwise we're waiting/wasting up to 2 min per job.

        '''

        # save output
        element = driver.find_element(By.XPATH, "//a[contains(@href,'tdump')]")
        href = element.get_attribute('href')
        url = 'https://www.ready.noaa.gov/%s'%href.split('\'')[1]
        JOB_ID = url.split('.')[-2]
        response = requests.get(url)
        # file_path = f'../output/tdump/{RF}_{UTC:05}_{WARD}_{JOB_ID}_tdump.txt'
        file_path = f'../output/tdump/{RF}_{UTC:05}_{WARD}_{ROUND}_{LEVEL}_{JOB_ID}_tdump.txt'
        with open(file_path, 'wb') as f: f.write(response.content)
        with open(file_path, 'r') as f: lines = f.readlines()
        output = lines[int(lines[0].split()[0])+1 + int(lines[int(lines[0].split()[0])+1].split()[0])+1+1:]
        output = ''.join(str(x) for x in output)
        os.system(f'cp ../output/output_template.txt ../output/{RF}_{UTC:05}_{WARD}_{ROUND}_{LEVEL}_{JOB_ID}_output.txt')
        with open(f'../output/{RF}_{UTC:05}_{WARD}_{ROUND}_{LEVEL}_{JOB_ID}_output.txt', 'a') as f:
            f.write(output)
        
        # Set paths
        src = '../input/input_template.txt'
        dst = f'../input/{RF}_{UTC:05}_{WARD}_{ROUND}_{LEVEL}_{JOB_ID}_input.txt'
        
        # Ensure the destination directory exists
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        
        # Copy the template file
        shutil.copy(src, dst)

        def write_input(file_path, line_num, var):
            with open(file_path, 'r') as f: lines = f.readlines()
            if 0 < line_num <= len(lines):
                lines[line_num - 1] = lines[line_num - 1].rstrip('\n')+str(var)+'\n'
            with open(file_path, 'w') as f: f.writelines(lines)
        fpath = dst
        write_input(fpath,  4, num_trajector_starting_locations)
        write_input(fpath,  5, type_of_trajectory)
        write_input(fpath, 10, meteorology)
        write_input(fpath, 12, coordinate_type)
        write_input(fpath, 13, latitude)
        write_input(fpath, 14, longitude)
        write_input(fpath, 19, trajectory_direction)
        write_input(fpath, 21, vertical_motion)
        write_input(fpath, 24, year)
        write_input(fpath, 25, month)
        write_input(fpath, 26, day)
        write_input(fpath, 27, hour)
        write_input(fpath, 29, total_run_time_hr)
        write_input(fpath, 30, new_trajectory_freq_hr)
        write_input(fpath, 31, max_num_trajectories)
        write_input(fpath, 33, auto_midboundary_layer_height)
        write_input(fpath, 34, altitude)
        write_input(fpath, 35, height_units)
        write_input(fpath, 40, GIS_output)
        write_input(fpath, 42, dpi)
        write_input(fpath, 43, zoom_factor)
        write_input(fpath, 44, projection)
        write_input(fpath, 46, vertical_units)
        write_input(fpath, 48, label_interval)
        write_input(fpath, 49, color_trajectories)
        write_input(fpath, 50, color_by_source)
        write_input(fpath, 51, source_symbol)
        write_input(fpath, 53, distance_overlay)
        write_input(fpath, 54, bounty_borders)
        write_input(fpath, 56, postscript_file)
        write_input(fpath, 57, pdf_file)
        write_input(fpath, 59, plot_met_field)
        write_input(fpath, 61, met_data_terrain_m)
        write_input(fpath, 62, met_data_potentialT_K)
        write_input(fpath, 63, met_data_ambientT_K)
        write_input(fpath, 64, met_data_rain_mm_hr)
        write_input(fpath, 65, met_data_mixed_layer_depth_m)
        write_input(fpath, 66, met_data_RH)
        write_input(fpath, 67, met_data_dn_solar_flux_w_m2)

        driver.close()
        print('HYSPLIT RUN COMPLETE            ', end='\r')
        print('\n\n')

# %%

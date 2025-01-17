# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 08:10:38 2024

@author: repooley
"""

import icartt
import pathlib
import pandas as pd
import matplotlib.pyplot as plt 

##--Gets the directory where this script file is located--##
wd = pathlib.Path(__file__).parent

##--Opens ICARTT file and creates a dataset object--##
ict = icartt.Dataset(r"C:\Users\repooley\NETCARE2015\ICARTT F‎iles\Flight3\AIMMS_POLAR6_20150408_R1_V1_NETCARE.txt")

##--Pulls data for each variable--##
temperature = ict.data["Temp"]
altitude = ict.data['Alt']
##--Humidity is converted to percent--##
humidity = ict.data['RH']*100
##--Pressure is converted from Pa to hPa--##
pressure = ict.data['BP']/100

###############
##--BINNING--##
###############

##--Creates a Pandas dataframe with all variables--##
df = pd.DataFrame({'Altitude': altitude, 'Humidity': humidity, 'Temperature': temperature, 'Pressure': pressure})

##--Define number of bins here--##
num_bins = 124

##--Pandas 'cut' splits altitude data into specified number of bins--##
df['Altitude_bin'] = pd.cut(df['Altitude'], bins=num_bins)

##--Group variables into each altitude bin--## 
##--Observed=false shows all bins, even empty ones--##
binned_df = df.groupby('Altitude_bin', observed=False).agg(
    
   ##--Aggregate data by mean, min, and max--##
    Humidity_avg=('Humidity', 'mean'),
    Humidity_min=('Humidity', 'min'),
    Humidity_max=('Humidity', 'max'),
    Humidity_25th=('Humidity', lambda x: x.quantile(0.25)),
    Humidity_75th=('Humidity', lambda x: x.quantile(0.75)),
    Temperature_avg=('Temperature', 'mean'),
    Temperature_min=('Temperature', 'min'),
    Temperature_max=('Temperature', 'max'),
    Temperature_25th=('Temperature', lambda x: x.quantile(0.25)),
    Temperature_75th=('Temperature', lambda x: x.quantile(0.75)),
    Pressure_avg=('Pressure', 'mean'),
    Pressure_min=('Pressure', 'min'),
    Pressure_max=('Pressure', 'max'),
    Pressure_25th=('Pressure', lambda x: x.quantile(0.25)),
    Pressure_75th=('Pressure', lambda x: x.quantile(0.75)),
    Altitude_center=('Altitude', 'mean')
    
    ##--Reset the index so Altitude_bin is just a column--##
).reset_index()

################
##--PLOTTING--##
################

##--Creates figure with 3 horizontally stacked subplots sharing a y-axis--##
fig, axs = plt.subplots(1, 3, figsize=(8, 6), sharey=True)

##--First subplot: Humidity vs Altitude--##

##--Averaged data in each bin is plotted against bin center--##
axs[0].plot(binned_df['Humidity_avg'], binned_df['Altitude_center'], color='navy', label='Humidity')
##--Range is given by filling between data minimum and maximum for each bin--##
axs[0].fill_betweenx(binned_df['Altitude_center'], binned_df['Humidity_25th'], binned_df['Humidity_75th'], color='lightsteelblue', alpha=0.1)
axs[0].fill_betweenx(binned_df['Altitude_center'], binned_df['Humidity_min'], binned_df['Humidity_max'], color='lightsteelblue', alpha=1)
axs[0].set_xlabel('Relative Humidity (%)')
axs[0].set_ylabel('Altitude (m)')
axs[0].set_title('Relative Humidity')
axs[0].set_xlim(0, 120)

##--Second subplot: Temperature vs Altitude--##
axs[1].plot(binned_df['Temperature_avg'], binned_df['Altitude_center'], color='darkslategray', label='Temperature')
axs[1].fill_betweenx(binned_df['Altitude_center'], binned_df['Temperature_min'], binned_df['Temperature_max'], color='cadetblue', alpha=0.6)
axs[1].fill_betweenx(binned_df['Altitude_center'], binned_df['Temperature_25th'], binned_df['Temperature_75th'], color='cadetblue', alpha=0.1)
axs[1].set_xlabel('Temperature (°C)')
axs[1].set_title('Temperature')
axs[1].set_xlim(-45, -16)

##--Third subplot: Pressure vs Altitude--##
axs[2].plot(binned_df['Pressure_avg'], binned_df['Altitude_center'], color='darkolivegreen', label='Pressure')
axs[2].fill_betweenx(binned_df['Altitude_center'], binned_df['Pressure_min'], binned_df['Pressure_max'], color='yellowgreen', alpha=0.5)
axs[2].set_xlabel('Pressure (hPa)')
axs[2].set_title('Pressure')
axs[2].set_xlim(390, 1040)

##--Gives the entire figure a title--##
plt.suptitle("Alert, NU 04/07/2015 Flight - Meteorological Vertical Profiles", fontsize=16)

##--Adjusts layout to prevent overlapping--## 
plt.tight_layout(rect=[0, 0, 1, 0.99])

##--Saves Figure--##
output_path = r"C:\Users\repooley\NETCARE2015\AltitudeBinnedData\Meteorological_Data_Altitude_Binned\Flight3"  
plt.savefig(output_path, dpi=300, bbox_inches='tight') 

plt.show()


# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 12:43:44 2024

@author: repooley
"""

import icartt
import pandas as pd
import matplotlib.pyplot as plt 

#########################
##--Open ICARTT Files--##
#########################

##--Meterological data from AIMMS monitoring system--##
aimms = icartt.Dataset(r"C:\Users\repooley\NETCARE2015\ICARTT F‎iles\AIMMS_POLAR6_20150407_R0_NETCARE.txt")

##--Carbon Monoxide Trace Gas--##
CO = icartt.Dataset(r"C:\Users\repooley\NETCARE2015\ICARTT F‎iles\CO_POLAR6_20150407_R1_NETCARE.txt")

##--Carbon Dioxide Trace Gas--##
CO2 = icartt.Dataset(r"C:\Users\repooley\NETCARE2015\ICARTT F‎iles\CO2_POLAR6_20150407_R1_NETCARE.txt")

##--Water Trace Gas--##
H2O = icartt.Dataset(r"C:\Users\repooley\NETCARE2015\ICARTT F‎iles\H2O_POLAR6_20150407_R1_NETCARE.txt")

##--First Half of Ozone Data--##
O3_1 = icartt.Dataset(r"C:\Users\repooley\NETCARE2015\ICARTT F‎iles\O3_POLAR6_201504071605_R0_L1_NETCARE.txt")

##--Second Half of Ozone Data--##
O3_2 = icartt.Dataset(r"C:\Users\repooley\NETCARE2015\ICARTT F‎iles\O3_POLAR6_201504071927_R0_L1_NETCARE.txt")

#################
##--Pull data--##
#################

##--AIMMS Data--##
altitude = aimms.data['Alt']
##--Humidity is converted to percent--##
humidity = aimms.data['RH']*100
aimms_time =aimms.data['TimeWave']

##--Trace Gas Data--##
CO_time = CO.data['Time_UTC']
CO_conc = CO.data['CO_ppbv']
CO2_time = CO2.data['Time_UTC']
CO2_conc = CO2.data['CO2_ppmv']
H2O_time = H2O.data['Time_UTC']
H2O_conc = H2O.data['H2O_ppmv']
O3_1_starttime = O3_1.data['Start_UTC']
O3_1_stoptime = O3_1.data['Stop_UTC']
O3_1_hour = O3_1.data['hr']
O3_1_conc = O3_1.data['O3']
O3_2_starttime = O3_2.data['Start_UTC']
O3_2_stoptime = O3_2.data['Stop_UTC']
O3_2_hour = O3_2.data['hr']
O3_2_conc = O3_2.data['O3']

###############
##--BINNING--##
###############

##--Creates a Pandas dataframe with all variables--##
df = pd.DataFrame({'Altitude': altitude, 'Humidity': humidity, 'CO_conc':CO_conc, 
                   'CO2_conc': CO2_conc, 'H2O_conc':H2O_conc})

##--Define number of bins here--##
num_bins = 124

##--Pandas 'cut' splits altitude data into specified number of bins--##
df['Altitude_bin'] = pd.cut(df['Altitude'], bins=num_bins)

##--Group variables into each altitude bin--## 
##--Observed=false shows all bins, even empty ones--##
binned_df = df.groupby('Altitude_bin', observed=False).agg(
    
   ##--Aggregate data by mean, min, and max--##
    Altitude_center=('Altitude', 'mean'),
    Humidity_avg=('Humidity', 'mean'),
    Humidity_min=('Humidity', 'min'),
    Humidity_max=('Humidity', 'max'),
    CO_conc_avg=('CO_conc', 'mean'),
    CO_conc_min=('CO_conc', 'min'),
    CO_conc_max=('CO_conc', 'max'),
    CO2_conc_avg=('CO2_conc', 'mean'),
    CO2_conc_min=('CO2_conc', 'min'),
    CO2_conc_max=('CO2_conc', 'max'),
    H2O_conc_avg=('H2O_conc', 'mean'),
    H2O_conc_min=('H2O_conc', 'min'),
    H2O_conc_max=('H2O_conc', 'max')
    
    ##--Reset the index so Altitude_bin is just a column--##
).reset_index()

################
##--PLOTTING--##
################

##--Creates figure with 3 horizontally stacked subplots sharing a y-axis--##
fig, axs = plt.subplots(1, 3, figsize=(9, 6), sharey=True)

##--First subplot: 10+ nm Particles vs Altitude--##

##--Averaged data in each bin is plotted against bin center--##
axs[0].plot(binned_df['CO_conc_avg'], binned_df['Altitude_center'], color='crimson', label='CO')
##--Range is given by filling between data minimum and maximum for each bin--##
axs[0].fill_betweenx(binned_df['Altitude_center'], binned_df['CO_conc_min'], 
                     binned_df['CO_conc_max'], color='palevioletred', alpha=0.3)
axs[0].set_xlabel('Concentration (ppbv)')
axs[0].set_ylabel('Altitude (m)')
axs[0].set_title('CO')
#axs[0].set_xlim(0, 1500)

##--Second subplot: 2.5+ nm Particles vs Altitude--##
axs[1].plot(binned_df['CO2_conc_avg'], binned_df['Altitude_center'], color='indigo', label='CO2')
axs[1].fill_betweenx(binned_df['Altitude_center'], binned_df['CO2_conc_min'], 
                     binned_df['CO2_conc_max'], color='darkorchid', alpha=0.3)
axs[1].set_xlabel('Concentration (ppmv)')
axs[1].set_title('CO\u2082')
#axs[1].set_xlim(0, 2000)

##--Third subplot: Pressure vs Altitude--##
axs[2].plot(binned_df['H2O_conc_avg'], binned_df['Altitude_center'], color='midnightblue', label='H2O')
axs[2].fill_betweenx(binned_df['Altitude_center'], binned_df['H2O_conc_min'], 
                     binned_df['H2O_conc_max'], color='mediumblue', alpha=0.2)
axs[2].set_xlabel('Concentration (ppmv)')
axs[2].set_title('H\u2082O')
axs[2].set_xlim(0, 1000)

##--Gives the entire figure a title--##
plt.suptitle("Alert, NU 04/07/2015 Flight - Vertical Profiles of Trace Gases", fontsize=16)

##--Adjusts layout to prevent overlapping--## 
plt.tight_layout(rect=[0, 0, 1, 0.99])

##--Saves Figure--##
output_path = "C:/Users/repooley/NETCARE2015/TimeBinnedData/flight2tracegas_profiles.png"  
plt.savefig(output_path, dpi=300, bbox_inches='tight') 

plt.show()

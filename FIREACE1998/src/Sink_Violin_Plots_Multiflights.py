# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 09:11:59 2025

@author: repooley
"""

import icartt
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.stats import mannwhitneyu

###################
##--User inputs--##
###################

##--Set the base directory to project folder--##
directory = r"C:\Users\repooley\REP_PhD\Arctic_NPF\FIREACE1998\data"

##--Choose which flights to analyze here!--##
flights_to_analyze = ["Flight3", "Flight7", 
                      'Flight9', 'Flight10', 
                      'Flight11', 'Flight12', 'Flight13', 
                      'Flight14', 'Flight15', 'Flight16', 
                      'Flight17', 'Flight18']

##--Set number of bins for latitude and potential temperature--##
num_bins_lat = 10
num_bins_ptemp = 10

##--Base output path for figures in directory--##
#output_path = r"C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data\processed\ViolinPlots"

PCASP_bins_path = r"C:\Users\repooley\REP_PhD\Arctic_NPF\FIREACE1998\data\raw\FIREACE1998_PCASP_bins.csv"

#########################
##--Open ICARTT Files--##
#########################

##--Define function that creates datasets from filenames--##
def find_files(directory, flight, partial_name):
    ##--flight data are stored in a folder called "raw"--##
    flight_dir = os.path.join(directory, "raw", flight)
    search_pattern = os.path.join(flight_dir, f"*{partial_name}*")
    return sorted(glob.glob(search_pattern))

###########################
##--Per-flight analysis--##
###########################

##--Store processed data for ALL flights here: --##
condensation_sinks = []
coagulation_sinks = []

##--Loop through each flight, pulling and analyzing data--##
for flight in flights_to_analyze:
    
    ##--Pull csv file containing all data--##
    files = find_files(directory, flight, "FIREACE")

    ##--The averaged data is always the second file--##
    if files:
        data = pd.read_csv(files[1])
    
    ##--Pull data variables from file--##
    time = data['Time'] # HHMMSS UTC time
    pressure = data['Pressure'] * 100 # in Pa
    temperature = data['Temperature'] + 273.15 # in K
    RH = data['RH'] # percent wrt water
    altitude = data['Altitude'] # in m (agl?)
    latitude = data['Latitude'] # degrees
    #longitude = data['Longitude'] # degrees

    ##--Particle data, 3 and 10 nm cutoffs, respectively--##
    CPC3_data = data['CN3025'] # Uncorrected data has a flow issue - but corrected not populated for many flights
    CPC10_data = data['CN7610']

    PCASP_bins = pd.read_csv(PCASP_bins_path)

    PCASP_data = data.iloc[:, 14:29] # select PCASP data

    ##--Add time, total_num to UHSAS_bins df--##
    PCASP_data.insert(0, 'Time', data['Time'])

    ##--Set time as the index for later alignment--##
    PCASP_data = PCASP_data.set_index('Time')

    ##--15 total bins--##
    PCASP_bin_num = [f'bin_{i}' for i in range(1, 16)]

    ##--Information for bins--##
    PCASP_bin_center = PCASP_bins['bin_avg']
    PCASP_lower_bound = PCASP_bins['lower_bound']
    PCASP_upper_bound = PCASP_bins['upper_bound']

    ##--Put column names and content in a dictionary and then convert to a Pandas df--##
    PCASP_df = pd.DataFrame({col: PCASP_data[col] for col in PCASP_bin_num})

    ##--Create new column names by rounding the bin center values to the nearest integer--##
    PCASP_new_col_names = PCASP_bin_center.round().astype(int).tolist()

    ##--Rename the PCASP_bins df columns to bin average values--##
    PCASP_data.columns = PCASP_new_col_names

    ##--Nans are denoted by -8888--##


    ######################
    ##--Calc N(2.5-10)--##
    ######################

    ##--Convert to STP!--##
    P_STP = 101325  # Pa
    T_STP = 273.15  # K

    ##--Create empty list for CPC3 particles--##
    CPC3_conc_STP = []

    for CPC3, T, P in zip(CPC3_data, temperature, pressure):
        if np.isnan(CPC3) or np.isnan(T) or np.isnan(P):
            ##--Append with NaN if any input is NaN--##
            CPC3_conc_STP.append(np.nan)
        else:
            ##--Perform conversion if all inputs are valid--##
            CPC3_conversion = CPC3 * (P_STP / P) * (T / T_STP)
            CPC3_conc_STP.append(CPC3_conversion)
        
    ##--Create empty list for CPC10 particles--##
    CPC10_conc_STP = []

    for CPC10, T, P in zip(CPC10_data, temperature, pressure):
        if np.isnan(CPC10) or np.isnan(T) or np.isnan(P):
            ##--Append with NaN if any input is NaN--##
            CPC10_conc_STP.append(np.nan)
        else:
            ##--Perform conversion if all inputs are valid--##
            CPC10_conversion = CPC10 * (P_STP / P) * (T / T_STP)
            CPC10_conc_STP.append(CPC10_conversion)

    ##--Creates a Pandas dataframe for particle data--##
    particle_df = pd.DataFrame({'Altitude': altitude, 'Latitude': latitude,
                       'CPC3_conc':CPC3_conc_STP, 'CPC10_conc': CPC10_conc_STP})

    ##--Calculate N3-10 particles--##
    nuc_particles = (particle_df['CPC3_conc'] - particle_df['CPC10_conc'])

    ##--Change calculated particle counts less than zero to NaN--##
    nuc_particles = np.where(nuc_particles >= 0, nuc_particles, np.nan)

    ##--Add nucleating particles to df--##
    particle_df['n_3_10'] = nuc_particles


    ############################
    ##--Normalize PCASP data--##
    ############################

    ##--Calculate dlogDp for each bin in numpy array--##
    dlogDp = np.log(PCASP_upper_bound.values) - np.log(PCASP_lower_bound.values)

    ##--Get only particle count data (excluding 'Time')--##
    PCASP_particle_counts = PCASP_data.loc[:, PCASP_new_col_names]

    ##--Normalize counts by dividing by dlogDp across all rows--##
    PCASP_dNdlogDp = PCASP_data.divide(dlogDp, axis=1)

    ##--Convert to STP!--##
    P_STP = 101325  # Pa
    T_STP = 273.15  # K

    ##--Create empty list for PCASP particles--##
    PCASP_STP = []

    for PCASP, T, P in zip(PCASP_dNdlogDp.values, data['Temperature']+273.15, data['Pressure']*100):
        if np.isnan(T) or np.isnan(P):
            ##--Append with NaN if any input is NaN--##
            PCASP_STP.append([np.nan]*len(PCASP))
        else:
            ##--Perform conversion if all inputs are valid--##
            corrected_PCASP = PCASP * (P_STP / P) * (T / T_STP)
            PCASP_STP.append(corrected_PCASP)

    ##--Convert back to DataFrame with same columns and index--##
    PCASP_STP = pd.DataFrame(PCASP_STP, columns=PCASP_dNdlogDp.columns, index=particle_df.index)

    ##--Add PCASP data to the dataframe--##
    particle_df = pd.concat([particle_df, PCASP_STP], axis=1)

    ##--Add PCASP total counts to the dataframe--##
    particle_df['PCTcon'] = data['PCTcon']

    ######################
    ##--Calc N(10-130)--##
    ######################

    ##--Calculate particles below UHSAS lower cutoff--##
    n_10_130 = (particle_df['CPC10_conc'] - particle_df['PCTcon'])

    ##--Change calculated particle counts less than zero to NaN--##
    n_10_130 = np.where(n_10_130 >= 0, n_10_130, np.nan)

    ##--Put N(10-130) bin center in a df--##
    n_10_130_center = pd.DataFrame([70])

    particle_df['n_10_130'] = n_10_130

    ##--Compute TOTAL counts from all size bins combined--##
    particle_df['Total_particles_STP'] = (particle_df['n_3_10'].fillna(0) + 
          particle_df['n_10_130'].fillna(0) + particle_df['PCTcon'].fillna(0))

    ###########################
    ##--Wrangle binned data--##
    ###########################

    ##--Concatenate bin edges--##
    combined_bin_edges = np.concatenate([
        [2.5],      # start of first bin
        [10],       # upper edge of N(2.5-10), also lower of next
        [130],       # upper edge of N(10-130), also lower of next
        PCASP_upper_bound.values,  # PCASP bins continue from 130
    ])

    time_averaged = data['Time']

    ##--Calculate time edges for each bin--##
    time_step = time_averaged.iloc[1] - time_averaged.iloc[0]  
    time_edges = np.append(time_averaged, time_averaged.iloc[-1] + time_step)  # length N + 1

    ##--Concatenate bin centers and reindex--##
    bin_centers = pd.concat([n_10_130_center, PCASP_bin_center], axis=0).reset_index(drop=True)

    ##--Place all binned data in a single df--##
    all_bins_aligned = PCASP_STP
    all_bins_aligned['6.25'] = particle_df['n_3_10']
    all_bins_aligned['70'] = particle_df['n_10_130']

    time_index = data['Time']  # use the same index as coagulation_sink

    ##--Ensure particle bin dataframes are indexed to time_index and properly named--##
    diameter_dfs = {}
    for col in all_bins_aligned.columns:
        vals = all_bins_aligned[col].to_numpy()
        # explicitly name the column as the bin diameter
        diameter_dfs[col] = pd.DataFrame({str(col): vals}, index=time_index[-len(vals):])
        
    #######################################
    ##--Calculate potential temperature--##
    #######################################

    ##--Constants--##
    p_0 = 1E5 # Reference pressure in Pa (1000 hPa)
    k = 0.286 # Poisson constant for dry air

    ##--Generate empty list for potential temperature output--##
    potential_temp = []

    ##--Calculate potential temperature from ambient temp & pressure--##
    for T, P in zip(temperature, pressure):
        p_t = T*(p_0/P)**k
        potential_temp.append(p_t)

    particle_df['PTemp'] = potential_temp

    
    ######################################
    ##--Condensation sink calculations--##
    ######################################

    ##--Constants--##

    R = 8.314 # Ideal gas constant (m^3*Pa*K^-1*mol^-1)
    ##--H2SO4 kinetic diam: lifted from Williamson et al for now (avg of their values)--##
    Ds = 5.49E-10 # in m
    ##--Kinetic diam of air calculated from mixing ratios and dataset on Wikepedia--##
    Dair = 3.61E-10 # in m
    avg_diam = (Ds + Dair)/2
    ##--Mass sulfuric acid--##
    Ms = 98.079 # g/mol
    ##--Mass air--##
    Mair = 28.96 # g/mol
    ##--Reduced mass--##
    Z = Ms/Mair 
    ##--Sticking coefficient - fair to assume unity for H2SO4--##
    alpha = 1
    ##--Boltzmann--##
    k = 1.38E-23 # J/K
    ##--Sutherland's law for dynamic viscosity--##
    C = 1.458E-6 # kg/ms*sqrt(K)
    S = 110.4 # K

    ##--Variables--##

    ##--Convert temperature and pressure from numpy array to dataframe to subvert errors--##
    ##--Convert temperature and pressure to series with aligned index--##
    temperature_series = pd.Series(temperature.values, index=time_index[-len(temperature):])
    pressure_series = pd.Series(pressure.values, index=time_index[-len(pressure):])

    Latitude_series = pd.Series(latitude, index=time_index)

    ##--Loop through dfs in diameter_dfs and calculate needed variables for each bin--##
    ##--Store in series initialized at zero--##
    condensation_sink = pd.Series(0, index=time_index)  

    for diameter, df in diameter_dfs.items():
        
        ##--Convert column diams from string to float--##
        mean_diameter = (float(diameter)) * 1E-9 # in m
        
        ##--Calculate mean free path of H2SO4 from molecular diameter--##
        df['mean_free_path'] = ((R * temperature_series.values) / ((2 ** (1/2)) 
                                * 3.14159 * (Ds ** 2) * 6.022E23 * pressure_series.values)) # in m/molecule
        
        ##--Calculate the Knudsen number--##
        df['Knudsen_num'] = df['mean_free_path'] / (mean_diameter / 2) # unitless ratio
        
        ##--Calculate Fuch's correction--##
        df['Fuchs_correction'] = (1 + df['Knudsen_num']) / (1 + ((4/(3*alpha)) 
                                + 0.337) * df['Knudsen_num'] + (4/(3*alpha) * (df['Knudsen_num']) ** 2)) # unitless
        
        ##--Calculate slip correction for Dynamic Viscosity calculation--##
        df['Slip_correction'] = (1 + df['Knudsen_num'] * (2.514 + 0.800 * (np.exp(-0.550 / df['Knudsen_num'])))) # unitless
        
        ##--Calculate dynamic viscosity using Sutherland's law with constants for air--##
        df['Dynamic_viscosity'] = (C * (temperature_series.values) ** (3/2)) / (temperature_series.values + S)
        
        ##--Calculate the diffusion coefficient--##
        df['Diffusion_coefficient'] = ((k * temperature_series.values * df['Slip_correction']) / 
                                       (3 * 3.14159 * df['Dynamic_viscosity'] * Ds)) # m^2/s
        
        ##--Extract Particle Concentration (first column in diameter_dfs)--##
        df['Particle_concentration'] = df.iloc[:, 0] / 1E-6 # converted to #/m^3
        
        ##--Per-bin contribution to condensation sink (before final multiplication)--##
        df['CS_contribution'] = (df['Fuchs_correction'] * mean_diameter * df['Particle_concentration'])
        
        ##--Multiply each bin’s CS contribution by its diffusion coefficient--##
        ##--Fill NaN values in CS_contribution with zeros to prevent NaN result--##
        condensation_sink += (2 * np.pi * df['Diffusion_coefficient'] * df['CS_contribution']).fillna(0)
     
    ##--Populate series--##
    condensation_sink = pd.DataFrame({'Condensation_Sink': condensation_sink, 'PTemp': potential_temp})
    
    #####################################
    ##--Coagulation sink calculations--##
    #####################################

    ##--Constants--##
    R = 8.314  # m^3*Pa*K^-1*mol^-1
    k = 1.38E-23  # m^2*kg*s^-2*K^-1
    C = 1.458E-6  # kg/ms*sqrt(K)
    S = 110.4  # K
    MMair = 28.96  # g/mol
    Mair = MMair/(6.022E23 * 1000)  # kg
    Dair = 3.61E-10  # m

    ##--For nucleation particles N(2.5-10)--##
    nuc_diam = 6.25E-9  # m
    nuc_vol = (4/3) * np.pi * (nuc_diam / 2) ** 3
    nuc_mass = nuc_vol
    z_nuc = nuc_mass / Mair
    sigma_nuc = (Dair + nuc_diam) / 2

    ##--Canonical time index from CSV--##
    time_index = data['Time']  # use as master index for all series

    ##--Ensure particle bin dataframes are indexed to time_index--##
    diameter_dfs = {}
    for col in all_bins_aligned.columns:
        vals = all_bins_aligned[col].to_numpy()
        # broadcast values to last N times of master index
        diameter_dfs[col] = pd.DataFrame({col: vals}, index=time_index[-len(vals):])

    ##--Convert temperature and pressure to series with aligned index--##
    temperature_series = pd.Series(temperature.values, index=time_index[-len(temperature):])
    pressure_series = pd.Series(pressure.values, index=time_index[-len(pressure):])

    Latitude_series = pd.Series(latitude, index=time_index)


    ##--Concentration of air molecules (number density)##
    Nair = (6.022E23 * pressure_series) / (R * temperature_series)  # num/m^3

    ##--Dynamic viscosity (Sutherland's law)##
    dynam_viscosity = (C * temperature_series ** (3/2)) / (temperature_series + S)

    ##--Mean speed of nucleation particles##
    nuc_speed = np.sqrt((8 * k * temperature_series) / (np.pi * nuc_mass))

    ##--Mean free path against air for slip correction##
    nuc_mfp_estimate = 1/(np.pi * np.sqrt(1 + z_nuc) * Nair * sigma_nuc**2)

    ##--Knudsen number and Cunningham slip correction##
    nuc_knudsen = nuc_mfp_estimate / (nuc_diam/2)
    nuc_slip = 1 + 2 * nuc_knudsen * (2.514 + 0.800 * np.exp(-0.550 / nuc_knudsen))

    ##--Diffusivity of nucleation particles##
    nuc_diffusivity = (k * temperature_series * nuc_slip / (3 * np.pi * dynam_viscosity * nuc_diam))

    ##--Mean free path of nucleation particles##
    nuc_mfp = (8 * nuc_diffusivity / (np.pi * nuc_speed))

    ##--g coefficient for nucleation particles##
    nuc_g = (np.sqrt(2) / (3 * nuc_diam * nuc_mfp)) * ((nuc_diam + nuc_mfp)**3 - (nuc_diam**2 + nuc_mfp**2)**(3/2)) - nuc_diam

    ##--Initialize coagulation sink series##
    coagulation_sink = pd.Series(0, index=time_index)

    #####################################
    ##--Loop through diameter bins--##
    #####################################
    for diameter, df in diameter_dfs.items():
        
        ##--Convert bin name to float diameter in meters##
        mean_diameter = float(diameter) * 1e-9  # m

        ##--Particle volume and mass (density = 1)##
        volume = (4/3) * np.pi * (mean_diameter / 2) ** 3
        mass = volume

        ##--Mean particle speed##
        speed = np.sqrt((8 * k * temperature_series) / (np.pi * mass))

        ##--Reduced mass ratio##
        z = mass / Mair

        ##--Collision cross section with air##
        sigma = (mean_diameter + Dair) / 2

        ##--Estimate mean free path against air for slip correction##
        mfp_estimate = 1 / (np.pi * np.sqrt(1 + z) * Nair * sigma**2)

        ##--Knudsen number and slip correction##
        knudsen_number = mfp_estimate / (mean_diameter / 2)
        slip = 1 + 2 * knudsen_number * (2.514 + 0.800 * np.exp(-0.550 / knudsen_number))

        ##--Particle diffusivity##
        diffusivity = (k * temperature_series * slip / (3 * np.pi * dynam_viscosity * mean_diameter))

        ##--Mean free path of H2SO4##
        mean_free_path = (8 * diffusivity / (np.pi * speed))

        ##--g coefficient##
        g = (np.sqrt(2) / (3 * mean_diameter * mean_free_path)) * \
            ((mean_diameter + mean_free_path)**3 - (mean_diameter**2 + mean_free_path**2)**(3/2)) - mean_diameter

        ##--Particle concentration from bin (converted to #/m^3)##
        df['Particle_concentration'] = df.iloc[:,0].fillna(0) / 1e-6

        ##--Coagulation kernel per bin##
        df['Coagulation_kernel'] = (2 * np.pi * (nuc_diffusivity + diffusivity) * (nuc_diam + mean_diameter) * 
            ((nuc_diam + mean_diameter) / (nuc_diam + mean_diameter + 2*np.sqrt(nuc_g**2 + g**2))
             + 8 * (nuc_diffusivity + diffusivity) / np.sqrt(nuc_speed**2 + speed**2) / (nuc_diam + mean_diameter))**-1)

        ##--Coagulation per bin##
        df['Coagulation'] = df['Coagulation_kernel'] * df['Particle_concentration']

        ##--Add to total coagulation sink (reindex to fill missing times with 0)##
        coagulation_sink += df['Coagulation'].reindex(coagulation_sink.index, fill_value=0)

    coagulation_sink = pd.DataFrame({'Coagulation': coagulation_sink, 'PTemp': potential_temp})
    
    #############################
    ##--Propagate uncertainty--##        
    #############################

    ##--Use the 75th quartile median uncertainty from all of NETCARE--##
    nuc_error_3sigma = 133.71
    
    ##--Create series to add--##
    nucleating_series = particle_df['n_3_10']
    lod_series = nuc_error_3sigma
    
    ##--Make safe copies before adding columns--##
    condensation_sink = condensation_sink.copy()
    coagulation_sink = coagulation_sink.copy()
    
    ##--Add the new columns--##
    condensation_sink['nucleating'] = nucleating_series.values

    condensation_sink['LoD'] = lod_series
    
    coagulation_sink['nucleating'] = nucleating_series.values
    coagulation_sink['LoD'] = lod_series
    
    coagulation_sinks.append(coagulation_sink)
    condensation_sinks.append(condensation_sink)
    

##--Concatenate the resulting lists of dataframes into single dataframes--##
condensation_sinks = pd.concat(condensation_sinks)
coagulation_sinks = pd.concat(coagulation_sinks)

#######################################
##--Filter to NPF and non-NPF times--##
#######################################

LoD = 133.71

condensation_npf = condensation_sinks.loc[condensation_sinks['nucleating'] > LoD, 'Condensation_Sink']
condensation_nonpf = condensation_sinks.loc[condensation_sinks['nucleating'] <= LoD, 'Condensation_Sink']

coagulation_npf = coagulation_sinks.loc[coagulation_sinks['nucleating'] > LoD, 'Coagulation']
coagulation_nonpf = coagulation_sinks.loc[coagulation_sinks['nucleating'] <= LoD, 'Coagulation']

##--Final dataframes to feed to the violin plots--##
##--Drop index to prevent reindexing issues--##
condensation = pd.DataFrame({'NPF': condensation_npf.reset_index(drop=True),
    'No NPF': condensation_nonpf.reset_index(drop=True)})

coagulation = pd.DataFrame({'NPF': coagulation_npf.reset_index(drop=True),
    'No NPF': coagulation_nonpf.reset_index(drop=True)})

##--Counts--##
conden_npf_count = len(condensation_npf)
conden_nonpf_count = len(condensation_nonpf)

coag_npf_count = len(coagulation_npf)
coag_nonpf_count = len(coagulation_nonpf)

################
##--Plotting--##
################

##--CONDENSATION--##

##--Order of label appearances:--##
group_order = ['NPF', 'No NPF']

##--Define color palette--##
palette = {'NPF': '#C00000', 'No NPF':'#820000'}

##--Use subplots for breaking y-axis--##
fig, (ax_top, ax_bottom) = plt.subplots(ncols=1, nrows=2, figsize=(6,8), sharex=True, 
                                        height_ratios=[1, 8], gridspec_kw={'hspace':0.08})

##--Cut=0 disallows interpolation beyond the data extremes--##
##--Set inner whisker length to zero for better clarity--##
sns.violinplot(data=condensation, order = ['NPF', 'No NPF'], 
                                   inner_kws={'whis_width': 0, 'solid_capstyle':'butt'}, palette=palette, ax=ax_top, cut=0)
##--Below the break: copy--##
sns.violinplot(data=condensation, order = ['NPF', 'No NPF'], 
                                   inner_kws={'whis_width': 0, 'solid_capstyle':'butt'}, palette=palette, ax=ax_bottom, cut=0)

##--Set limits above and below the break--##
ax_top.set_ylim(0.0008, 0.004) 
ax_bottom.set_ylim(-0.00005, 0.0008)

##--Remove duplicated spines--##
sns.despine(ax=ax_bottom, right=False)
sns.despine(ax=ax_top, bottom=True, right=False, top=False)

##--Add diagonal break lines--##

ax = ax_top
ax2 = ax_bottom
##--length of break lines--##
d = .015  
##--Top diagonal--##
ax.plot((-d, +d), (-d, +d), transform=ax_top.transAxes, color='k', clip_on=False)
##--Bottom diagonal--##
##--Bottom break — adjust d to match top angle (scale by inverse of height ratio)--##
d_scaled = d * (1 / 8)
ax2.plot((-d, +d), (1 - d_scaled, 1 + d_scaled), transform=ax_bottom.transAxes, color='k', clip_on=False) 

fig.supylabel('Condensation Sink $(S^{-1})$', fontsize=14, x=-0.05)

plt.suptitle('Condensation Sink', fontsize=14, y=0.92)

ax.set(xlabel='')
ax.set_xticks(range(len(group_order)))
ax.set_xticklabels(group_order)

##--Add secondary x-axis labels for high and low lat regions--##
fig.supxlabel('65-75\u00b0N', fontsize=14, x=0.32, y=0.045)
plt.text(0.64, 0.045, '>75\u00b0N', transform=fig.transFigure, fontsize=14)

ax_top.tick_params(axis='x', which='both', labelsize=14, top=False, labeltop=False)
ax_top.tick_params(axis='y', which='both', labelsize=14, top=False, labeltop=False)
ax_bottom.tick_params(axis='x', which='both', labelsize=14, top=False, labeltop=False)
ax_bottom.tick_params(axis='y', which='both', labelsize=14, top=False, labeltop=False)

##--Add text labels with N--##
plt.text(0.26, 0.125, "N={}".format(conden_npf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')
plt.text(0.64, 0.125, "N={}".format(conden_nonpf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')

#plt.savefig(f"{output_path}\\Sinks\\condensation\conden_MultiFlight", dpi=600)

plt.show()

##--COAGULATION--##

palette2 = {'NPF': '#EA1010', 'No NPF':'#9d0b0b'}

##--Use subplots for breaking y-axis--##
fig, (ax_top, ax_bottom) = plt.subplots(ncols=1, nrows=2, figsize=(6,8), sharex=True, 
                                        height_ratios=[1, 8], gridspec_kw={'hspace':0.08})

##--Cut=0 disallows interpolation beyond the data extremes--##
##--Set inner whisker length to zero for better clarity--##
sns.violinplot(data=coagulation, order = ['NPF', 'No NPF'], 
                                   inner_kws={'whis_width': 0, 'solid_capstyle':'butt'}, palette=palette2, ax=ax_top, cut=0)
##--Below the break: copy--##
sns.violinplot(data=coagulation, order = ['NPF', 'No NPF'], 
                                   inner_kws={'whis_width': 0, 'solid_capstyle':'butt'}, palette=palette2, ax=ax_bottom, cut=0)

##--Set limits above and below the break--##
ax_top.set_ylim(0.0002, 0.0025) 
ax_bottom.set_ylim(-0.00001, 0.0002)

##--Remove duplicated spines--##
sns.despine(ax=ax_bottom, right=False)
sns.despine(ax=ax_top, bottom=True, right=False, top=False)

##--Add diagonal break lines--##

ax = ax_top
ax2 = ax_bottom
##--length of break lines--##
d = .015  
##--Top diagonal--##
ax.plot((-d, +d), (-d, +d), transform=ax_top.transAxes, color='k', clip_on=False)
##--Bottom diagonal--##
##--Bottom break — adjust d to match top angle (scale by inverse of height ratio)--##
d_scaled = d * (1 / 8)
ax2.plot((-d, +d), (1 - d_scaled, 1 + d_scaled), transform=ax_bottom.transAxes, color='k', clip_on=False) 

fig.supylabel('Coagulation Sink $(S^{-1})$', fontsize=14, x=-0.1)


plt.suptitle('Coagulation Sink', fontsize=14, y=0.92)

ax.set(xlabel='')
ax.set_xticks(range(len(group_order)))
ax.set_xticklabels(group_order)

##--Add secondary x-axis labels for high and low lat regions--##
fig.supxlabel('65-75\u00b0N', fontsize=14, x=0.32, y=0.045)
plt.text(0.64, 0.045, '>75\u00b0N', transform=fig.transFigure, fontsize=14)

ax_top.tick_params(axis='x', which='both', labelsize=14, top=False, labeltop=False)
ax_top.tick_params(axis='y', which='both', labelsize=14, top=False, labeltop=False)
ax_bottom.tick_params(axis='x', which='both', labelsize=14, top=False, labeltop=False)
ax_bottom.tick_params(axis='y', which='both', labelsize=14, top=False, labeltop=False)

##--Add text labels with N--##
plt.text(0.26, 0.125, "N={}".format(coag_npf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')
plt.text(0.64, 0.125, "N={}".format(coag_nonpf_count), transform=fig.transFigure, fontsize=10, color='dimgrey')


#plt.savefig(f"{output_path}\\Sinks\\coagulation\coag_MultiFlight", dpi=600)

plt.show()
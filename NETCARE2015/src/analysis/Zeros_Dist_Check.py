# -*- coding: utf-8 -*-
"""
Created on Tue May 12 15:31:58 2026

@author: repooley
"""


import icartt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.stats import mannwhitneyu

#####################################
##--Calculate std dev from zeroes--##        
#####################################

##--Pull datasets with zeros not filtered out--##
##--Worth it to do flight by flight or no?--##
CPC3_R1 = icartt.Dataset(r"C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data\raw\CPC_R1\CPC3776_Polar6_20150408_R1_L2.ict")    
CPC10_R1 = icartt.Dataset(r'C:\Users\repooley\REP_PhD\Arctic_NPF\NETCARE2015\data\raw\CPC_R1\CPC3772_Polar6_20150408_R1_L2.ict')
CPC3_R1_conc = CPC3_R1.data['conc']
CPC10_R1_conc = CPC10_R1.data['conc']

##--Isolate zero periods, setting conservative upper limit of 50 for pulling these out--##
##--The zero periods are actually all counts between 1-6--##
##--Numpy doesn't recognize -9999 as NaN, tell it to ignore these values--##
CPC3_zeros_c = CPC3_R1_conc[(CPC3_R1_conc < 50) & (CPC3_R1_conc != -9999)]
CPC10_zeros_c = CPC10_R1_conc[(CPC10_R1_conc < 50) & (CPC10_R1_conc != -99999)]

##--Calculate standard deviation of zeros--##
CPC3_sigma = np.std(CPC3_zeros_c, ddof=1)  # Use ddof=1 for sample standard deviation
CPC10_sigma = np.std(CPC10_zeros_c, ddof=1)


plt.hist(CPC3_zeros_c, bins=100, color='skyblue', edgecolor='none')
plt.xlim(0, 10)
plt.xlabel('Counts with Filter')
plt.ylabel('Frequency')
plt.title('CPC3 Zero Distribution')
plt.show()


plt.hist(CPC10_zeros_c, bins=50, color="Purple", edgecolor='none')
plt.xlim(0, 10)
plt.xlabel('Counts with Filter')
plt.ylabel('Frequency')
plt.title('CPC10 Zero Distribution')
plt.show()




#!/usr/bin/env python
# coding: utf-8

# In[1]
import numpy as np
from statsmodels.tsa.stattools import adfuller,kpss
import pandas as pd
from matplotlib import pyplot as plt
import torch
import tigramite
from tigramite import data_processing as pp
from tigramite.toymodels import structural_causal_processes as toys
from tigramite import plotting as tp
from tigramite.lpcmci import LPCMCI
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from statsmodels.tools.sm_exceptions import InterpolationWarning
from tigramite.independence_tests.cmiknn import CMIknn
from tigramite.lpcmci import LPCMCI
import seaborn as sns
# from tigramite.independence_tests.gpdc import GPDC
# from tigramite.independence_tests.cmiknn import CMIknn
# from tigramite.independence_tests.cmisymb import CMIsymb


# Load Data, Handle Missingness:

data = torch.load(f"/home/gnicolaou/tigramite/tutorials/causal_discovery/combined_tensor.pt")

# Number of observed variables
N = data.shape[1]
dat = data.numpy()

# empty list to store the modified columns to separate trajectories
modified_columns = []

for col in range(dat.shape[1]):
    column_data = dat[:, col]
    
    # insert 999 values between every 144 elements
    modified_column = np.insert(column_data, np.arange(144, column_data.size, 144), 999)
    
    # append the modified column to the list
    modified_columns.append(modified_column)

modified_data = np.column_stack(modified_columns)

dat = modified_data # handle missingness
n_a_n = np.isnan(dat).any(axis=1)
dat[n_a_n] = 999
#dat = dat[160:292,:]

# initialize dataframe object, specify variable names
var_names = ['Nd','Pr','sst','lts','fth','ws','div','cf']
dataframe = pp.DataFrame(dat, var_names=var_names, missing_flag = 999)


parcorr = ParCorr(significance='analytic')

#cmi_knn = CMIknn(significance='fixed_thres', model_selection_folds=3)
print(0)
link_assumptions = {j:{(i, -tau):'' for i in range(8) for tau in range(2) if (i, -tau) != (j, 0)} 
                            for j in range(8)}

link_assumptions[0][(1, 0)] = '<?-' #Nd is an ancestor of P
link_assumptions[1][(0, 0)] = '-?>'
link_assumptions[7][(1, 0)] = '<?-'# CF is an ancestor of precipitation
link_assumptions[1][(7, 0)] = '-?>'
link_assumptions[0][(0, -1)] = '-?>' #Nd at lag t-1 is an ancestor of Nd
link_assumptions[0][(1, -1)] = '-?>' #P at lag t-1 is an ancestor of Nd
link_assumptions[7][(1, -1)] = '-?>' #P at lag t-1 is an ancestor of CF
link_assumptions[7][(0, -1)] = 'o?>' #Nd at lag t-1 is an ancestor of CF
link_assumptions[1][(0, -1)] = 'o?>' #Nd at lag t-1 is an ancestor of P
link_assumptions[1][(7, -1)] = 'o?>' #CF at lag -1 is an ancestor of P
link_assumptions[7][(7, -1)] = 'o?>' #CF at lag t-1 is an ancestor of CF
for j in range(2,7): 
    link_assumptions[j][(0, 0)] = '<?-' #meterological variables are ancestor of aerosol
    link_assumptions[0][(j, 0)] = '-?>'
    link_assumptions[j][(7, 0)] = '<?-' #meteorological variables are ancestor of cloud fraction
    link_assumptions[7][(j, 0)] = '-?>'
#    link_assumptions[j][(j, -1)] = '-?>' #meterological variables at lag t-1 is an ancestor of meteorological variables
#    for k in range(2,7):
#        if (j != k):
#            link_assumptions[j][k,0] = 'o?o' #all meteorology variables are contemporaneously linked
#            link_assumptions[k][j,0] = 'o?o'
#            link_assumptions[j][k,-1] = 'o?>' #all meteorology variables at lag t-1 is an ancestor of eachother
#            link_assumptions[k][j,-1] = 'o?>' 

print(link_assumptions)

pcmci_parcorr = LPCMCI(
    dataframe=dataframe, 
    cond_ind_test=parcorr,
    verbosity=1)
results = pcmci_parcorr.run_lpcmci(tau_max=2, pc_alpha = .05, link_assumptions= link_assumptions)
print(results)
tp.plot_time_series_graph(graph=results['graph'],
                          val_matrix=results['val_matrix'], save_name = "parcorr_linkass_noM_.05.png")
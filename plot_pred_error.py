import time
import argparse

import numpy as np

from scipy import stats
from scipy.stats import linregress

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.available
mpl.style.use('seaborn-paper') 

result1 = np.load('results_det/000_pred.npy')                  
result2 = np.load('results/000_pred.npy') 
result3 = np.load('results_back/000_pred.npy') 

#pcl = np.load('results_pcl/000_pred.npy') 
  
#nopcl_relu = np.load('results_flow_nopcl_relu/000_pred.npy')              



#fig = plt.figure(figsize=(6.1,6.1))
fig = plt.figure(figsize=(11.1,6.1))

#plt.plot(np.mean(nopcl_relu, axis=0), 'o--', lw=3, label='Physics-agnostic AE (ReLU)', color='#31a354')            
#plt.fill_between(x=range(nopcl_relu.shape[1]),y1=np.mean(nopcl_relu, axis=0)-np.var(nopcl_relu, axis=0)**0.5, y2=np.mean(nopcl_relu, axis=0)+np.var(nopcl_relu, axis=0)**0.5, color='#31a354', alpha=0.2)         
##plt.fill_between(x=range(nopcl.shape[1]), y1=np.quantile(nopcl, .01, axis=0), y2=np.quantile(nopcl, .99, axis=0), color='#377eb8', alpha=0.2)         
#  

plt.plot(np.mean(result1, axis=0), 'o--', lw=3, label='Dynamic AE', color='#2c7fb8')            
plt.fill_between(x=range(result1.shape[1]),y1=np.mean(result1, axis=0)-np.var(result1, axis=0)**0.5, y2=np.mean(result1, axis=0)+np.var(result1, axis=0)**0.5, color='#2c7fb8', alpha=0.2)         
#plt.fill_between(x=range(nopcl.shape[1]), y1=np.quantile(nopcl, .01, axis=0), y2=np.quantile(nopcl, .99, axis=0), color='#377eb8', alpha=0.2)         
          
plt.plot(np.mean(result2, axis=0), 'o--', lw=3, label='Variational Dynamic AE', color='#31a354')            
plt.fill_between(x=range(result2.shape[1]), y1=np.mean(result2, axis=0)-np.var(result2, axis=0)**0.5, y2=np.mean(result2, axis=0)+np.var(result2, axis=0)**0.5, color='#31a354', alpha=0.2)         
#plt.fill_between(x=range(pcl.shape[1]), y1=np.quantile(pcl, .01, axis=0), y2=np.quantile(pcl, .99, axis=0), color='#f03b20', alpha=0.2)                   
                 
plt.plot(np.mean(result3, axis=0), 'o--', lw=3, label='Variational Dynamic AE (backwards)', color='#f03b20')            
plt.fill_between(x=range(result3.shape[1]), y1=np.mean(result3, axis=0)-np.var(result3, axis=0)**0.5, y2=np.mean(result3, axis=0)+np.var(result3, axis=0)**0.5, color='#f03b20', alpha=0.2)         
#plt.fill_between(x=range(pcl.shape[1]), y1=np.quantile(pcl, .01, axis=0), y2=np.quantile(pcl, .99, axis=0), color='#f03b20', alpha=0.2)         
             
                 
                 
plt.tick_params(axis='x', labelsize=22) 
plt.tick_params(axis='y', labelsize=22) 
plt.tick_params(axis='both', which='minor', labelsize=22)

#plt.locator_params(axis='y', nbins=6)
#plt.locator_params(axis='x', nbins=6)


plt.ylabel('Relative prediction error', fontsize=22)
plt.xlabel('Time step', fontsize=22)
plt.grid(False)
#plt.yscale("log")
maxmax = np.maximum(result1.max(), result2.max())
#plt.ylim([0.01, 10**-0.1])
plt.legend(fontsize=19, loc="upper left")
fig.tight_layout()  
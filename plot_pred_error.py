import time
import argparse

import numpy as np

from scipy import stats
from scipy.stats import linregress

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.available
mpl.style.use('seaborn-paper') 









result1 = np.load('results_det_pendulum_4/000_pred.npy') 
result2 = np.load('results_det_back_pendulum_4/000_pred.npy')


result1 = np.load('results_det_pendulum_4_noise/000_pred.npy') 
result2 = np.load('results_det_back_pendulum_4_noise/000_pred.npy')



result1 = np.load('results_det_pendulum_lin_2/000_pred.npy')                  
result2 = np.load('results_det_back_pendulum_lin_2/000_pred.npy') 


result1 = np.load('results_det_pendulum_lin_2_noise/000_pred.npy')                  
result2 = np.load('results_det_back_pendulum_lin_2_noise/000_pred.npy') 

result1 = np.load('results_det_flow/000_pred.npy')                  
result2 = np.load('results_det_back_flow/000_pred.npy') 

result1 = np.load('results_det_flow_alpha2/000_pred.npy')                  
result2 = np.load('results_det_back_flow_alpha2/000_pred.npy') 

#result1 = np.load('results_det_sphere_s2_ns/000_pred.npy')                  
#result2 = np.load('results_det_back_sphere_s2_ns/000_pred.npy') 
#result3 = np.load('results_var_sphere_s2_ns/000_pred.npy') 
#result4 = np.load('results_var_back_sphere_s2_ns/000_pred.npy') 




#pcl = np.load('results_pcl/000_pred.npy') 
  
#nopcl_relu = np.load('results_flow_nopcl_relu/000_pred.npy')              



#fig = plt.figure(figsize=(6.1,6.1))
fig = plt.figure(figsize=(11.1,6.1))

#plt.plot(np.mean(nopcl_relu, axis=0), 'o--', lw=3, label='Physics-agnostic AE (ReLU)', color='#31a354')            
#plt.fill_between(x=range(nopcl_relu.shape[1]),y1=np.mean(nopcl_relu, axis=0)-np.var(nopcl_relu, axis=0)**0.5, y2=np.mean(nopcl_relu, axis=0)+np.var(nopcl_relu, axis=0)**0.5, color='#31a354', alpha=0.2)         
##plt.fill_between(x=range(nopcl.shape[1]), y1=np.quantile(nopcl, .01, axis=0), y2=np.quantile(nopcl, .99, axis=0), color='#377eb8', alpha=0.2)         
#  

plt.plot(np.mean(result1, axis=0), 'o-', lw=2, label='DAE', color='#2c7fb8')            
plt.fill_between(x=range(result1.shape[1]), y1=np.percentile(result1, 10, axis=0), y2=np.percentile(result1, 90, axis=0), color='#2c7fb8', alpha=0.2)         
#plt.fill_between(x=range(nopcl.shape[1]), y1=np.quantile(nopcl, .01, axis=0), y2=np.quantile(nopcl, .99, axis=0), color='#377eb8', alpha=0.2)         
          
plt.plot(np.mean(result2, axis=0), 'o-', lw=2, label='Bijective DAE', color='#c51b8a')            
plt.fill_between(x=range(result2.shape[1]), y1=np.percentile(result2, 10, axis=0), y2=np.percentile(result2, 90, axis=0), color='#c51b8a', alpha=0.2)         
#plt.fill_between(x=range(pcl.shape[1]), y1=np.quantile(pcl, .01, axis=0), y2=np.quantile(pcl, .99, axis=0), color='#f03b20', alpha=0.2)                   
                 
plt.plot(np.mean(result3, axis=0), 'o--', lw=3, label='DAE', color='#f03b20')            
plt.fill_between(x=range(result3.shape[1]), y1=np.percentile(result3, 10, axis=0), y2=np.percentile(result3, 90, axis=0), color='#f03b20', alpha=0.2)         
#plt.fill_between(x=range(pcl.shape[1]), y1=np.quantile(pcl, .01, axis=0), y2=np.quantile(pcl, .99, axis=0), color='#f03b20', alpha=0.2)         
             
plt.plot(np.mean(result4, axis=0), 'o--', lw=3, label='Bijective DAE', color='#31a354')            
plt.fill_between(x=range(result4.shape[1]), y1=np.percentile(result4, 10, axis=0), y2=np.percentile(result4, 90, axis=0), color='#31a354', alpha=0.2)         
#plt.fill_between(x=range(pcl.shape[1]), y1=np.quantile(pcl, .01, axis=0), y2=np.quantile(pcl, .99, axis=0), color='#f03b20', alpha=0.2)         
             
                        
                 
plt.tick_params(axis='x', labelsize=24) 
plt.tick_params(axis='y', labelsize=24) 
plt.tick_params(axis='both', which='minor', labelsize=24)

#plt.locator_params(axis='y', nbins=6)
#plt.locator_params(axis='x', nbins=6)

plt.ylabel('Relative prediction error', fontsize=24)
plt.xlabel('Time step', fontsize=24)
plt.grid(False)
#plt.yscale("log")
maxmax = np.maximum(result1.max(), result2.max())
plt.ylim([0.000, 0.3])
#plt.ylim([0.000, 0.08])
plt.ylim([0.000, 0.1])
plt.legend(fontsize=24, loc="lower right")
plt.legend(fontsize=24, loc="upper left")

fig.tight_layout()  
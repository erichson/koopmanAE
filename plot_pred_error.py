import time
import argparse

import numpy as np

from scipy import stats
from scipy.stats import linregress

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.available
mpl.style.use('seaborn-paper') 


from matplotlib.ticker import StrMethodFormatter


result1 = np.load('results_pendulum/000_pred.npy')                  
result2 = np.load('results_back_pendulum/000_pred.npy')   

    
def moving_average(a, n=4) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n    
    

fig = plt.figure(figsize=(12,4))           

plt.plot(np.mean(result1, axis=0), '-', lw=2, label='Koopman AE', color='#377eb8')            
plt.fill_between(x=range(result1.shape[1]), y1=np.mean(result1, axis=0)-np.var(result1, axis=0)**0.5, y2=np.mean(result1, axis=0)+np.var(result1, axis=0)**0.5, color='#377eb8', alpha=0.2)         

plt.plot(np.mean(result2, axis=0), '-', lw=2, label='Consistent Koopman AE', color='#e41a1c')            
plt.fill_between(x=range(result2.shape[1]), y1=np.mean(result2, axis=0)-np.var(result2, axis=0)**0.5, y2=np.mean(result2, axis=0)+np.var(result2, axis=0)**0.5, color='#e41a1c', alpha=0.2)         
  
                 
plt.tick_params(axis='x', labelsize=18) 
plt.tick_params(axis='y', labelsize=18) 
plt.tick_params(axis='both', which='minor', labelsize=16)
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places

plt.xlabel('time, t',fontsize=18)
plt.ylabel('prediction error',fontsize=18)

plt.grid(False)
maxmax = np.maximum(result1.max(), result2.max())
plt.legend(fontsize=18, loc="upper left")
fig.tight_layout()  
plt.show()
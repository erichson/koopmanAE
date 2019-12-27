import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

from matplotlib import pylab as plt
from scipy.special import ellipj, ellipk

#******************************************************************************
# Read in data
#******************************************************************************
def data_from_name(name):
    if name == 'harmonic':
        return harmonic()
    if name == 'pendulum':
        return pendulum()    
    if name == 'flow':
        return flow_cylinder()
    if name == 'flow_noisy':
        return flow_cylinder_noisy()   
    if name == 'flow150':
        return flow_cylinder150()    
    
    else:
        raise ValueError('dataset {} not recognized'.format(name))


def rescale(Xsmall, Xsmall_test):
    #******************************************************************************
    # Rescale data
    #******************************************************************************
    Xmin = Xsmall.min()
    Xmax = Xsmall.max()
    
    Xsmall = ((Xsmall - Xmin) / (Xmax - Xmin)) 
    Xsmall_test = ((Xsmall_test - Xmin) / (Xmax - Xmin)) 

    return Xsmall, Xsmall_test



def harmonic():
    
    np.random.seed(1234567899)
    
    # Create dataset
    def F(w,t):
        return np.array( [[np.cos(w*t), np.sin(w*t)], [-np.sin(w*t), np.cos(w*t)]] )


    X = []
    x0 = np.array([0,1])

    for i in np.arange(0.0, 1000, 0.5):
        X.append(F(0.1,i).dot(x0) + np.random.standard_normal(2) * 0.01 )
        
    X = np.asarray(X)
    
    # Rotate to high-dimensional space
    Q = np.random.standard_normal((144,2))
    Q,_ = np.linalg.qr(Q)
    
    X = X.dot(Q.T) # rotate
    
    # scale 
    #X = 2 * (X - np.min(X)) / np.ptp(X) - 1
    
    # split into train and test set
    X_test = X[500:]
    X = X[:500]     

    #******************************************************************************
    # Return train and test set
    #******************************************************************************
    return X, X_test, 144, 1



def pendulum():
    
    np.random.seed(1234567899)

    def sol(t,theta0):
        S = np.sin(0.5*(theta0) )
        K_S = ellipk(S**2)
        omega_0 = np.sqrt(9.81)
        sn,cn,dn,ph = ellipj( K_S - omega_0*t, S**2 )
        theta = 2.0*np.arcsin( S*sn )
        d_sn_du = cn*dn
        d_sn_dt = -omega_0 * d_sn_du
        d_theta_dt = 2.0*S*d_sn_dt / np.sqrt(1.0-(S*sn)**2)
        return np.stack([theta, d_theta_dt],axis=1)
    
    
    
    anal_ts = np.linspace(0,15,1550)
    X = sol(anal_ts, 0.1)
    X = X.T
    X += np.random.standard_normal(X.shape) * 0.0
    
    
    #******************************************************************************
    # Empedding
    #******************************************************************************
    fig = plt.figure(figsize=(15,15))
    plt.plot(X[0,:], X[1,:], '-', lw=1, label='', color='#377eb8')
    
    plt.tick_params(axis='x', labelsize=22)
    plt.tick_params(axis='y', labelsize=22)
    plt.locator_params(axis='y', nbins=10)
    plt.locator_params(axis='x', nbins=10)
    
    plt.ylabel('Relative prediction error', fontsize=22)
    plt.xlabel('Time step', fontsize=22)
    plt.grid(False)
    #plt.yscale("log")
    #plt.legend(fontsize=22)
    fig.tight_layout()
    #plt.savefig(args.folder +'/000prediction' +'.eps')
    
    
    # Rotate to high-dimensional space
    Q = np.random.standard_normal((144,2))
    Q,_ = np.linalg.qr(Q)
    
    X = X.T.dot(Q.T) # rotate
    
    # scale 
    #X = 2 * (X - np.min(X)) / np.ptp(X) - 1
    
    # split into train and test set
    X_test = X[500:]
    X = X[:500]     

    #******************************************************************************
    # Return train and test set
    #******************************************************************************
    return X, X_test, 144, 1



def flow_cylinder():
    X = np.load('data/flow_cylinder.npy')
    print(X.shape)
    
    # Split into train and test set
    X = X[:, 65:, :]
    X = X[:, ::6, ::3]
    X = X[:, 0:64, 0:64]

    t, m, n = X.shape
 
    # mean subtract
    X = X.reshape(-1,m*m)
    X -= X.mean(axis=0)        
    
    # scale 
    X = X.reshape(-1,m*m)
    X = 2 * (X - np.min(X)) / np.ptp(X) - 1
    X = X.reshape(-1,m,m)    
    
    # split into train and test set
    
    X = X[0:151]   
    X_test = X[0:151]
    
    
    #******************************************************************************
    # Return train and test set
    #******************************************************************************
    return X, X_test, m, n


  
def flow_cylinder_noisy():
    X = np.load('data/flow_cylinder.npy')
    print(X.shape)
    
    # Split into train and test set
    X = X[:, 65:, :]
    X = X[:, ::6, ::3]
    X = X[:, 0:64, 0:64]

    t, m, n = X.shape
 
    signal = np.var(X)
    noise = signal / 120
    X = X + np.random.normal(0, noise**0.5, X.shape) * 1.0

    
    # scale 
    X = X.reshape(-1,m*m)
    X = 2 * (X - np.min(X)) / np.ptp(X) - 1
    X = X.reshape(-1,m,m) 
    
    # split into train and test set
    
    X_test = X[100:]
    X = X[:100]    
    
    
    #******************************************************************************
    # Return train and test set
    #******************************************************************************
    return X, X_test, m, n


def flow_cylinder150():
    from scipy.io import loadmat
    X = loadmat('data/sphere_s2_ns.mat')
    X = X['U']
    print(X.shape)
    
    # Split into train and test set
    X = X[:, 65:, :]
    X = X[:, ::6, ::3]
    X = X[:, 0:64, 0:64]

    t, m, n = X.shape
 
    # mean subtract
    X = X.reshape(-1,m*m)
    X -= X.mean(axis=0)        
    
    # scale 
    X = X.reshape(-1,m*m)
    X = 2 * (X - np.min(X)) / np.ptp(X) - 1
    X = X.reshape(-1,m,m)    
    
    # split into train and test set
    
    X = X[0:151]   
    X_test = X[0:151]
    
    
    #******************************************************************************
    # Return train and test set
    #******************************************************************************
    return X, X_test, m, n





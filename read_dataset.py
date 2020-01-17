import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

from matplotlib import pylab as plt
from scipy.special import ellipj, ellipk

from scipy.special import ellipj, ellipk


#******************************************************************************
# Read in data
#******************************************************************************
def data_from_name(name):
    if name == 'harmonic':
        return harmonic()
    if name == 'pendulum_lin':
        return pendulum_lin()      
    if name == 'pendulum':
        return pendulum()    
    if name == 'flow':
        return flow_cylinder()
    if name == 'flow_noisy':
        return flow_cylinder_noisy()   
    if name == 'sphere_s2_ns':
        return sphere_s2_ns()    
    
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
    X = X.T  
    
    # Rotate to high-dimensional space
    Q = np.random.standard_normal((144,2))
    Q,_ = np.linalg.qr(Q)
    
    X = X.T.dot(Q.T) # rotate
    
    # scale 
    #X = 2 * (X - np.min(X)) / np.ptp(X) - 1
    
    # split into train and test set
    X_train = X[:500]   
    X_test = X[500:]

    #******************************************************************************
    # Return train and test set
    #******************************************************************************
    return X_train, X_test, 144, 1


def pendulum_lin():
    
    np.random.seed(12346789)

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
    
    
    
    anal_ts = np.linspace(0,35,1650)
    X = sol(anal_ts, 0.785398)
    X = X.T
    Xclean = X.copy()
    X += np.random.standard_normal(X.shape) * 0.00
    

#    #******************************************************************************
#    # Empedding
#    #******************************************************************************
#    fig = plt.figure(figsize=(6,6))
#    plt.plot(Xclean[0,:], Xclean[1,:], '-', lw=3, color='k', label='True trajecotry')
#    plt.plot(X[0,:300], X[1,:300], 'o', lw=1, color='#dd1c77', label='Sampled data points', alpha=0.4)
#
#    plt.tick_params(axis='x', labelsize=18)
#    plt.tick_params(axis='y', labelsize=18)
#    plt.locator_params(axis='y', nbins=5)
#    plt.locator_params(axis='x', nbins=5)
#    
#    plt.ylabel('Velocity', fontsize=18)
#    plt.xlabel('Theta', fontsize=18)
#    plt.grid(False)
#    #plt.ylim(-2.5,2.5)
#    #plt.xlim(-2.5,2.5)
#    #plt.yscale("log")
#    #plt.legend(fontsize=18)
#    plt.axhline(y=0, color='k')
#    plt.axvline(x=0, color='k')      
#    plt.axis('off')
#    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')    
#            
#    
#    fig.tight_layout()
#    #plt.savefig(args.folder +'/000prediction' +'.eps')
#    
#    
#    #******************************************************************************
#    # Empedding
#    #******************************************************************************
#    fig = plt.figure(figsize=(8,3))
#    plt.plot(np.arange(1,602), Xclean[0,0:601], '-', lw=2, label='training', color='k')
#    plt.plot(np.arange(600,Xclean.shape[1]), Xclean[0,600::], '--', lw=2, label='test', color='k')
#    plt.plot(np.arange(0,Xclean.shape[1])[::5], X[0,::5], 'o', lw=1, color='#dd1c77', label='Sampled data points', alpha=0.4)
#        
#    plt.tick_params(axis='x', labelsize=22)
#    plt.tick_params(axis='y', labelsize=22)
#    plt.locator_params(axis='y', nbins=5)
#    plt.locator_params(axis='x', nbins=5)
#    
#    plt.ylabel('Theta', fontsize=22)
#    plt.xlabel('Time step', fontsize=22)
#    plt.grid(False)
#    plt.axhline(y=0, color='k')
#    plt.axvline(x=0, color='k')      
#    plt.axis('off')
#    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')    
#    
#    fig.tight_layout()
#    #plt.savefig(args.folder +'/000prediction' +'.eps')
#    
#    #******************************************************************************
#    # Empedding
#    #******************************************************************************
#    fig = plt.figure(figsize=(8,3))
#    plt.plot(np.arange(1,602), Xclean[1,0:601], '-', lw=2, label='training', color='k')
#    plt.plot(np.arange(600,Xclean.shape[1]), Xclean[1,600::], '--', lw=2, label='test', color='k')
#    plt.plot(np.arange(0,Xclean.shape[1])[::5], X[1,::5], 'o', lw=1, color='#dd1c77', label='Sampled data points', alpha=0.4)
#    
#    plt.tick_params(axis='x', labelsize=22)
#    plt.tick_params(axis='y', labelsize=22)
#    plt.locator_params(axis='y', nbins=5)
#    plt.locator_params(axis='x', nbins=5)
#    
#    plt.ylabel('Velocity', fontsize=22)
#    plt.xlabel('Time step', fontsize=22)
#    plt.grid(False)
#    plt.axhline(y=0, color='k')
#    plt.axvline(x=0, color='k')      
#    plt.axis('off')
#    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')    
#    
#    fig.tight_layout()
#    #plt.savefig(args.folder +'/000prediction' +'.eps')
                
    
    
    
    
    
    # Rotate to high-dimensional space
    Q = np.random.standard_normal((144,2))
    Q,_ = np.linalg.qr(Q)
    
    X = X.T.dot(Q.T) # rotate
    
    # scale 
    #X = 2 * (X - np.min(X)) / np.ptp(X) - 1
    
    # split into train and test set 
    X_train = X[:600]   
    X_test = X[600:]
    
    
    #******************************************************************************
    # Return train and test set
    #******************************************************************************
    return X_train, X_test, 144, 1


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
    
    
    anal_ts = np.linspace(0,15,2050)
    
    anal_ts = np.linspace(0,35,1650)
    X = sol(anal_ts, 2.35619)
    X = X.T
    Xclean = X.copy()
    X += np.random.standard_normal(X.shape) * 0.000
    
    
    # Rotate to high-dimensional space
    Q = np.random.standard_normal((144,2))
    Q,_ = np.linalg.qr(Q)
    
    X = X.T.dot(Q.T) # rotate
    
    # scale 
    #X = 2 * (X - np.min(X)) / np.ptp(X) - 1
    
    # split into train and test set 
    X_train = X[:600]   
    X_test = X[600:]
    
    #******************************************************************************
    # Return train and test set
    #******************************************************************************
    return X_train, X_test, 144, 1



def flow_cylinder():
    X = np.load('data/flow_cylinder.npy')
    print(X.shape)
    
    # Split into train and test set
    X = X[:, 65:, :]
    X = X[:, ::6, ::3]
    X = X[:, 0:64, 0:64]

    t, m, n = X.shape
 
    # mean subtract
    X = X.reshape(-1,m*n)
    X -= X.mean(axis=0)        
    
    # scale 
    X = X.reshape(-1,m*n)
    X = 2 * (X - np.min(X)) / np.ptp(X) - 1
    X = X.reshape(-1,m,n)    
    
    # split into train and test set
    
    X_train = X[30::]  
    X_test = X
    
    
    #******************************************************************************
    # Return train and test set
    #******************************************************************************
    return X_train, X_test, m, n


  
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
    X = X.reshape(-1,m*n)
    X = 2 * (X - np.min(X)) / np.ptp(X) - 1
    X = X.reshape(-1,m,n) 
    
    # split into train and test set
    
    X_train = X[30::]  
    X_test = X
    
    
    #******************************************************************************
    # Return train and test set
    #******************************************************************************
    return X_train, X_test, m, n


def sphere_s2_ns():
    from scipy.io import loadmat
    X = loadmat('data/sphere_s2_ns_x3.mat')
    X = X['U']
    print(X.shape)
    
    # Split into train and test set
    #X = X[::10, :]

    X = X.T
    t, m = X.shape
 
    # mean subtract
    X = X.reshape(-1,m)
    X -= X.mean(axis=0)        
    
    # normalize
    #X /= X.var(axis=0)**0.5       
    
    # scale 
    X = X.reshape(-1,m*1)
    X = 2 * (X - np.min(X)) / np.ptp(X) - 1
    #X = X.reshape(-1,m,1)    
    
    # split into train and test set
    
    X_train = X[50:] 
    X_test = X[10:]
    
    
    #******************************************************************************
    # Return train and test set
    #******************************************************************************
    return X_train, X_test, m, 1





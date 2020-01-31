import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

from matplotlib import pylab as plt
from scipy.special import ellipj, ellipk

from scipy.special import ellipj, ellipk

import torch

#******************************************************************************
# Read in data
#******************************************************************************
def data_from_name(name, noise = 0.0):
    if name == 'harmonic':
        return harmonic()
    if name == 'pendulum_lin':
        return pendulum_lin(noise)      
    if name == 'pendulum':
        return pendulum(noise)    
    if name == 'flow':
        return flow_cylinder()
    if name == 'flow_noisy':
        return flow_cylinder_noisy()   
    if name == 'sphere_s2_ns':
        return sphere_s2_ns()    
    if name == 'sphere_s2_tf':
        return sphere_s2_tf()      
    if name == 'sst':
        return sst()
        return sphere_s2_ns()
    if name == 'word_PTB':
        from torchtext.datasets import PennTreebank
        import torchtext
        return word_ptb()
    if name == 'char_PTB':
        from torchtext.datasets import PennTreebank
        import torchtext        
        return char_ptb()
    
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


def word_ptb():
    return PennTreebank.splits(text_field=torchtext.data.Field(batch_first=True))


def char_ptb():
    CHARS = torchtext.data.Field(tokenize=list)
    train, test, _ = torchtext.datasets.PennTreebank.splits(CHARS)
    CHARS.build_vocab(train)
    train_iter, test_iter = torchtext.data.BucketIterator.splits(
        (train, test), batch_size=1)
    num_classes = 50
    train = list(train_iter)[0].text - 2
    test = list(test_iter)[0].text - 2
    train = torch.nn.functional.one_hot(train.type(torch.int64),
                                        num_classes=num_classes).type(torch.FloatTensor).numpy()
    test = torch.nn.functional.one_hot(test.type(torch.int64),
                                       num_classes=num_classes).type(torch.FloatTensor).numpy()
    return train, test, 1, 50


def ptb():
    return PennTreebank.splits(text_field=torchtext.data.Field(batch_first=True))


def harmonic():
    
    np.random.seed(1)
    
    # Create dataset
    def F(w,t):
        return np.array( [[np.cos(w*t), np.sin(w*t)], [-np.sin(w*t), np.cos(w*t)]] )


    X = []
    x0 = np.array([0,1])

    for i in np.linspace(0,35,1650):
        X.append(F(1,i).dot(x0) + np.random.standard_normal(2) * 0.0 )
        
    X = np.asarray(X)
    X = X.T  
    
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
    return X_train, X_test, 64, 1





def pendulum_lin(noise):
    
    np.random.seed(1)

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
    
    
    
    anal_ts = np.arange(0, 2200*0.03, 0.03)
    
    X = sol(anal_ts, 0.7853981633974483)
    
    X = X.T
    Xclean = X.copy()
    X += np.random.standard_normal(X.shape) * noise
    

    #******************************************************************************
    # Empedding
    #******************************************************************************
    fig = plt.figure(figsize=(6,6))
    plt.plot(Xclean[0,:], Xclean[1,:], '-', lw=3, color='k', label='True trajecotry')
    plt.plot(X[0,:200], X[1,:200], 'o', lw=1, color='#dd1c77', label='Sampled data points', alpha=0.4)

    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='x', nbins=5)
    
    plt.ylabel('Velocity', fontsize=18)
    plt.xlabel('Theta', fontsize=18)
    plt.grid(False)
    #plt.ylim(-2.5,2.5)
    #plt.xlim(-2.5,2.5)
    #plt.yscale("log")
    #plt.legend(fontsize=18)
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')      
    plt.axis('off')
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')    
            
    
    fig.tight_layout()
    #plt.savefig(args.folder +'/000prediction' +'.eps')
    
    
    #******************************************************************************
    # Empedding
    #******************************************************************************
    fig = plt.figure(figsize=(8,3))
    plt.plot(np.arange(0,200), Xclean[0,0:200], '-', lw=2, label='training', color='k')
    #plt.plot(np.arange(600,Xclean.shape[1]), Xclean[0,600::], '--', lw=2, label='test', color='k')
    plt.plot(np.arange(0,Xclean.shape[1])[0:200], X[0,0:200], 'o', lw=1, color='#dd1c77', label='Sampled data points', alpha=0.4)
        
    plt.tick_params(axis='x', labelsize=22)
    plt.tick_params(axis='y', labelsize=22)
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='x', nbins=5)
    
    plt.ylabel('Theta', fontsize=22)
    plt.xlabel('Time step', fontsize=22)
    plt.grid(False)
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')      
    plt.axis('off')
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')    
    
    fig.tight_layout()
    #plt.savefig(args.folder +'/000prediction' +'.eps')
    
    #******************************************************************************
    # Empedding
    #******************************************************************************
    fig = plt.figure(figsize=(8,3))
    plt.plot(np.arange(0,200), Xclean[1,0:200], '-', lw=2, label='training', color='k')
    #plt.plot(np.arange(600,Xclean.shape[1]), Xclean[0,600::], '--', lw=2, label='test', color='k')
    plt.plot(np.arange(0,Xclean.shape[1])[0:200], X[1,0:200], 'o', lw=1, color='#dd1c77', label='Sampled data points', alpha=0.4)
       
    
    plt.tick_params(axis='x', labelsize=22)
    plt.tick_params(axis='y', labelsize=22)
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='x', nbins=5)
    
    plt.ylabel('Velocity', fontsize=22)
    plt.xlabel('Time step', fontsize=22)
    plt.grid(False)
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')      
    plt.axis('off')
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')    
    
    fig.tight_layout()
    #plt.savefig(args.folder +'/000prediction' +'.eps')
                
    
    
    plt.close('all')
    
    
    # Rotate to high-dimensional space
    Q = np.random.standard_normal((64,2))
    Q,_ = np.linalg.qr(Q)
    
    X = X.T.dot(Q.T) # rotate
    # mean subtract
    #X -= X.mean(axis=0)        
        
    
    Xclean = Xclean.T.dot(Q.T)
    # mean subtract
    #Xclean -= Xclean.mean(axis=0)        
    
    
    # scale 
    X = 2 * (X - np.min(X)) / np.ptp(X) - 1
    Xclean = 2 * (Xclean - np.min(Xclean)) / np.ptp(Xclean) - 1

    
    # split into train and test set 
    X_train = X[0:600]   
    X_test = X[600:]

    X_train_clean = Xclean[0:600]   
    X_test_clean = Xclean[600:]    
    
    #******************************************************************************
    # Return train and test set
    #******************************************************************************
    return X_train, X_test, X_train_clean, X_test_clean, 64, 1






def pendulum(noise):
    
    np.random.seed(1)

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
    
    
    anal_ts = np.arange(0, 2200*0.03, 0.03)
    X = sol(anal_ts, 2.4)
    
    X = X.T
    Xclean = X.copy()
    X += np.random.standard_normal(X.shape) * noise
    
    
    # Rotate to high-dimensional space
    Q = np.random.standard_normal((64,2))
    Q,_ = np.linalg.qr(Q)
    
    X = X.T.dot(Q.T) # rotate
    
    Xclean = Xclean.T.dot(Q.T)
    
    # scale 
    X = 2 * (X - np.min(X)) / np.ptp(X) - 1
    Xclean = 2 * (Xclean - np.min(Xclean)) / np.ptp(Xclean) - 1

    
    # split into train and test set 
    X_train = X[0:600]   
    X_test = X[600:]

    X_train_clean = Xclean[0:600]   
    X_test_clean = Xclean[600:]     
    
    #******************************************************************************
    # Return train and test set
    #******************************************************************************
    return X_train, X_test, X_train_clean, X_test_clean, 64, 1




def flow_cylinder():
    X = np.load('data/flow_cylinder.npy')
    print(X.shape)
    
    # Split into train and test set
    X = X[:, 65:, :]
    X = X[:, ::2, ::1]
    
    #X = X[:, ::6, ::3]
    #X = X[:, 0:64, 0:64]

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
    return X_train, X_test, X_train, X_test, m, n


  
def flow_cylinder_noisy():
    X = np.load('data/flow_cylinder.npy')
    print(X.shape)
    
    # Split into train and test set
    X = X[:, 65:, :]
    #X = X[:, ::6, ::3]
    #X = X[:, 0:64, 0:64]
    X = X[:, ::2, ::1]

    t, m, n = X.shape
 
    signal = np.var(X)
    noise = signal / 20
    X = X + np.random.normal(0, noise**0.5, X.shape) * 1.0


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
    
    
    X = np.load('data/flow_cylinder.npy')
    
    # Split into train and test set
    X = X[:, 65:, :]
    #X = X[:, ::6, ::3]
    #X = X[:, 0:64, 0:64]
    X = X[:, ::2, ::1]

    t, m, n = X.shape

    # mean subtract
    X = X.reshape(-1,m*n)
    X -= X.mean(axis=0)    
    
    # scale 
    X = X.reshape(-1,m*n)
    X = 2 * (X - np.min(X)) / np.ptp(X) - 1
    X = X.reshape(-1,m,n) 
    
    # split into train and test set
    
    X_train_clean = X[30::]  
    X_test_clean = X
        
    
    #plt.imshow(X_test[0], cmap=cmocean.cm.balance)
    
    #******************************************************************************
    # Return train and test set
    #******************************************************************************
    return X_train, X_test, X_train_clean, X_test_clean, m, n


def sst():

    X = np.load('data/sstday.npy')
    #******************************************************************************
    # Preprocess data
    #******************************************************************************
    t, m, n = X.shape


    #******************************************************************************
    # Slect train data
    #******************************************************************************
    #indices = np.random.permutation(1400)
    indices = range(3600)
    #training_idx, test_idx = indices[:730], indices[730:1000] 
    training_idx, test_idx = indices[220:1315], indices[1315:2557] # 6 years
    #training_idx, test_idx = indices[230:2420], indices[2420:2557] # 6 years
    #training_idx, test_idx = indices[0:1825], indices[1825:2557] # 5 years    
    #training_idx, test_idx = indices[230:1325], indices[1325:2000] # 3 years
    
    
    # mean subtract
    X = X.reshape(-1,m*n)
    X -= X.mean(axis=0)    
    
    # scale 
    X = X.reshape(-1,m*n)
    X = 2 * (X - np.min(X)) / np.ptp(X) - 1
    X = X.reshape(-1,m,n) 
    
    # split into train and test set
    
    X_train = X[training_idx]  
    X_test = X[test_idx]

 
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

def sphere_s2_tf():
    from scipy.io import loadmat
    X = loadmat('data/sphere_s2_tf.mat')
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



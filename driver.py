import time
import argparse

import numpy as np

from scipy import stats
from scipy.stats import linregress

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.available
mpl.style.use('seaborn-paper')


import torch
from torch import nn
from torch import autograd
from torch import optim
from torchvision import transforms, datasets
from torch.autograd import grad

from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable, Function


from timeit import default_timer as timer

import torch.nn.init as init

from read_dataset import data_from_name
from model import *
from tools import *

from train import *

#import ristretto as ro
#from ristretto.svd import compute_rsvd

import os

from scipy import stats

from Adam_new import *

# python driver.py --model net --dataset harmonic --epochs 400 --batch 32 --folder results --lamb 4.0 --steps 5


class shallow_re(nn.Module):
    def __init__(self, m, n, b):
        super(shallow_re, self).__init__()
        self.decoder = decoderNet(m, n, b)

    def forward(self, x):
        out = self.decoder(x)
        return out

#==============================================================================
# Training settings
#==============================================================================
parser = argparse.ArgumentParser(description='PyTorch Example')
#
parser.add_argument('--model', type=str, default='net', metavar='N', help='Model')
#
parser.add_argument('--dataset', type=str, default='harmonic', metavar='N', help='dataset')
#
parser.add_argument('--lr', type=float, default=1e-2, metavar='N', help='learning rate (default: 0.01)')
#
parser.add_argument('--wd', type=float, default=0.0, metavar='N', help='weight_decay (default: 1e-5)')
#
parser.add_argument('--epochs', type=int, default=600, metavar='N', help='number of epochs to train (default: 10)')
#
parser.add_argument('--batch', type=int, default=64, metavar='N', help='batch size (default: 10000)')
#
parser.add_argument('--batch_test', type=int, default=50, metavar='N', help='batch size  for test set (default: 10000)')
#
parser.add_argument('--plotting', type=bool, default=True, metavar='N', help='number of epochs to train (default: 10)')
#
parser.add_argument('--folder', type=str, default='results_det',  help='specify directory to print results to')
#
parser.add_argument('--lamb', type=float, default='4',  help='PCL penalty lambda hyperparameter')
#
parser.add_argument('--gamma', type=float, default='0',  help='Depricated')
#
parser.add_argument('--steps', type=int, default='3',  help='steps for omega')
#
parser.add_argument('--bottleneck', type=int, default='2',  help='bottleneck')
#
parser.add_argument('--lr_update', type=int, nargs='+', default=[100, 300, 500], help='Decrease learning rate at these epochs.')
#
parser.add_argument('--lr_decay', type=float, default='0.2',  help='PCL penalty lambda hyperparameter')
#
parser.add_argument('--pred_steps', type=int, default='1000',  help='Prediction steps')
#
parser.add_argument('--seed', type=int, default='1',  help='Prediction steps')
#


args = parser.parse_args()



set_seed()
device = get_device()



#******************************************************************************
# Create folder to save results
#******************************************************************************
if not os.path.isdir(args.folder):
    os.mkdir(args.folder)



#******************************************************************************
# load data
#******************************************************************************
Xtrain, Xtest, m, n = data_from_name(args.dataset)
Xfull = np.concatenate((Xtrain,Xtest))

#******************************************************************************
# Reshape data for pytorch into 4D tensor Samples x Channels x Width x Hight
#******************************************************************************
Xtrain = add_channels(Xtrain)
Xtest = add_channels(Xtest)
Xfull = add_channels(Xfull)

# transfer to tensor
Xtrain = torch.from_numpy(Xtrain).float().contiguous()
Xtest = torch.from_numpy(Xtest).float().contiguous()
Xfull = torch.from_numpy(Xfull).float().contiguous()


#******************************************************************************
# Create Dataloader objects
#******************************************************************************


trainDat = []
start = 0
for i in np.arange(args.steps,-1, -1):
    if i == 0:
        trainDat.append(Xtrain[start:].float())
    else:
        trainDat.append(Xtrain[start:-i].float())
    start += 1

train_data = torch.utils.data.TensorDataset(*trainDat)
train_loader = DataLoader(dataset = train_data,
                          batch_size = args.batch,
                          shuffle = True)


testDat = []
start = 0
for i in np.arange(args.steps,-1, -1):
    if i == 0:
        testDat.append(Xtest[start:].float())
    else:
        testDat.append(Xtest[start:-i].float())
    start += 1

test_data = torch.utils.data.TensorDataset(*testDat)
test_loader = DataLoader(dataset = test_data,
                          batch_size = args.batch_test,
                          shuffle = False)


del(trainDat, testDat)


#==============================================================================
# Model
#==============================================================================
print(Xtrain.shape)
if(args.model == 'net'):
    model = shallow_autoencoder(Xtrain.shape[2], Xtrain.shape[3], args.bottleneck, args.steps)
    model.apply(weights_init)
    print('net')

#elif(args.model == 'big'):
#    model = big_autoencoder(Xtrain.shape[2], Xtrain.shape[3])
#    model.apply(weights_init)
#    print('big')

model = torch.nn.DataParallel(model)


#==============================================================================
# Model summary
#==============================================================================
print('**** Setup ****')
print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
print('************')
print(model)


#==============================================================================
# Start training
#==============================================================================
model, optimizer, error_train, error_test, epoch_hist = train(model, train_loader, test_loader,
                    lr=args.lr, weight_decay=args.wd, lamb=args.lamb, num_epochs = args.epochs,
                    learning_rate_change=args.lr_decay, epoch_update=args.lr_update,
                    gamma = args.gamma)


#with open(args.folder+"/model.pkl", "wb") as f:
#    torch.save(model,f)
torch.save(model.state_dict(), args.folder + '/model'+'.pkl')


for param_group in optimizer.param_groups:
    print(param_group['weight_decay'])
    print(param_group['weight_decay_adapt'])

error_train = np.asarray(error_train)
error_test = np.asarray(error_test)
epoch_hist = np.asarray(epoch_hist)
np.save(args.folder +'/000_error_train.npy', error_train)
np.save(args.folder +'/000_error_test.npy', error_test)
np.save(args.folder +'/000_epoch_hist.npy', epoch_hist)




#******************************************************************************
# Error plots
#******************************************************************************
if(args.plotting == True):
        fig = plt.figure(figsize=(15,12))
        plt.plot(epoch_hist[:], error_train[:], lw=3, label='Train error', color='#377eb8',)
        plt.plot(epoch_hist[:], error_test[:], lw=3, label='Test error', color='#e41a1c',)


        plt.tick_params(axis='x', labelsize=22)
        plt.tick_params(axis='y', labelsize=22)
        plt.locator_params(axis='y', nbins=10)
        plt.locator_params(axis='x', nbins=10)

        plt.ylabel('Type II Error', fontsize=22)
        plt.xlabel('Epochs', fontsize=22)
        plt.grid(False)
        plt.yscale("log")
        #ax[0].set_ylim([0.01,1])
        plt.legend(fontsize=22)
        fig.tight_layout()
        plt.show()
        plt.savefig(args.folder +'/train_test_error' +'.png')
        #plt.savefig(args.folder +'/train_test_error' +'.eps')

        plt.close()







#******************************************************************************
# Prediction
#******************************************************************************
Xinput, Xtarget = Xtest[:-1], Xtest[1:]



error = []
for i in range(30):
            error_temp = []
            
            z = model.module.encoder(Xinput[i].float().to(device)) # embedd data in latent space


            for j in range(args.pred_steps):
                z = model.module.dynamics(z) # evolve system in time
                x_pred = model.module.decoder(z) # map back to high-dimensional space
                target_temp = Xtarget[i+j].data.cpu().numpy().reshape(m,n)
                error_temp.append(np.linalg.norm(x_pred.data.cpu().numpy().reshape(m,n) - target_temp) / np.linalg.norm(target_temp))

            error.append(np.asarray(error_temp))

error = np.asarray(error)

fig = plt.figure(figsize=(15,12))
plt.plot(error.mean(axis=0), 'o--', lw=3, label='', color='#377eb8')
plt.fill_between(x=range(error.shape[1]),y1=np.quantile(error, .05, axis=0), y2=np.quantile(error, .95, axis=0), color='#377eb8', alpha=0.2)

plt.tick_params(axis='x', labelsize=22)
plt.tick_params(axis='y', labelsize=22)
plt.locator_params(axis='y', nbins=10)
plt.locator_params(axis='x', nbins=10)

plt.ylabel('Relative prediction error', fontsize=22)
plt.xlabel('Time step', fontsize=22)
plt.grid(False)
#plt.yscale("log")
plt.ylim([0.0,error.max()*2])
#plt.legend(fontsize=22)
fig.tight_layout()
plt.savefig(args.folder +'/000prediction' +'.png')
#plt.savefig(args.folder +'/000prediction' +'.eps')

plt.close()

np.save(args.folder +'/000_pred.npy', error)





#******************************************************************************
# Empedding
#******************************************************************************
Xinput, Xtarget = Xtest[:-1], Xtest[1:]

emb = []
            
z = model.module.encoder(Xinput[i].float().to(device)) # embedd data in latent space

for j in range(args.pred_steps):
    z = model.module.dynamics(z) # evolve system in time
    emb.append(z.data.cpu().numpy().reshape(args.bottleneck))              

emb = np.asarray(emb)

fig = plt.figure(figsize=(15,15))
plt.plot(emb[:,0], emb[:,1], '-', lw=1, label='', color='#377eb8')

plt.xlim(-1.6, 1.6)
plt.ylim(-1.6, 1.6)

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
plt.savefig(args.folder +'/embedding' +'.png')
#plt.savefig(args.folder +'/000prediction' +'.eps')

plt.close()




#******************************************************************************
# Eigenvalues
#******************************************************************************
model.eval()
A =  model.module.dynamics.dynamics.weight.cpu().data.numpy()
#A =  model.module.test.data.cpu().data.numpy()
w, v = np.linalg.eig(A)
print(np.abs(w))

fig = plt.figure(figsize=(6.1, 6.1), facecolor="white",  edgecolor='k', dpi=150)
plt.scatter(w.real, w.imag, c = '#dd1c77', marker = 'o', s=15*6, zorder=2, label='Eigenvalues')

maxeig = 1.4
plt.xlim([-maxeig, maxeig])
plt.ylim([-maxeig, maxeig])
plt.locator_params(axis='x',nbins=4)
plt.locator_params(axis='y',nbins=4)

plt.xlabel('Real', fontsize=22)
plt.ylabel('Imaginary', fontsize=22)
plt.tick_params(axis='y', labelsize=22)
plt.tick_params(axis='x', labelsize=22)
plt.axhline(y=0,color='#636363',ls='-', lw=3, zorder=1 )
plt.axvline(x=0,color='#636363',ls='-', lw=3, zorder=1 )

#plt.legend(loc="upper left", fontsize=16)
t = np.linspace(0,np.pi*2,100)
plt.plot(np.cos(t), np.sin(t), ls='-', lw=3, c = '#636363', zorder=1 )
plt.tight_layout()
plt.show()
plt.savefig(args.folder +'/000eigs' +'.png')
plt.savefig(args.folder +'/000eigs' +'.eps')
plt.close()

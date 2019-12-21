#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 15:18:58 2019
test
@author: ben
"""

import torch
import torchvision
from torch import nn
from torch import autograd
from torch import optim
from torchvision import transforms, datasets
from torch.autograd import grad
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable, Function

import torch.nn.init as init

import numpy as np

from tools import *

from Adam_new import *

#from stable_projection import *


def train(model, train_loader, test_loader, lr, weight_decay, 
          lamb, num_epochs, learning_rate_change, epoch_update, gamma=0.0):

    #optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
    device = get_device()

    alpha_list = []
    for idx, (name, param) in enumerate(model.named_parameters()):
                if param.requires_grad:
                    if 'dynamics' not in name:
                        alpha_list.append(idx)            
            
        
    optimizer = AdamPCL(model.parameters(), lr=lr, 
                                weight_decay=weight_decay, 
                                weight_decay_adapt=0.0, 
                                alpha_list=alpha_list)    
                          
                
            
    def exp_lr_scheduler(optimizer, epoch, lr_decay_rate=0.8, decayEpoch=[]):
                    """Decay learning rate by a factor of lr_decay_rate every lr_decay_epoch epochs"""
                    if epoch in decayEpoch:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= lr_decay_rate
                        return optimizer
                    else:
                        return optimizer
                        
                
    def wd_scheduler(optimizer, weight_decay):
                """Decay learning rate by a factor of lr_decay_rate every lr_decay_epoch epochs"""
                for param_group in optimizer.param_groups:
                    param_group['weight_decay_adapt'] = weight_decay
                return optimizer  





                     
        
    criterion = nn.MSELoss().to(device)
    #criterion2 = nn.L1Loss().cuda()



    epoch_hist = []
    loss_hist = []
    epoch_loss = []


                            
    for epoch in range(num_epochs):
            
        for batch_idx, data_list in enumerate(train_loader):            
    
            model.train()
            out, out_back = model(data_list[0].to(device))
            
            for k in range(len(data_list)-1):
                if k == 0:
                    loss_pred = criterion(out[k], data_list[k+1].to(device))
                else:
                    loss_pred += criterion(out[k], data_list[k+1].to(device))

            
#            for k in range(1):
#                if k == 0:
#                    loss_back = criterion(out_back[k], data_list[k])
#                else:
#                    loss_back += criterion(out_back[k], data_list[k])
            
            
            
            loss_identity = criterion(out[len(data_list)-1], data_list[0].to(device))
            loss = loss_pred + lamb * loss_identity #+  0 * loss_back



            # AB = I and BA = I
#            A = model.module.dynamics.dynamics.weight
#            B = model.module.backdynamics.dynamics.weight
#            AB = torch.mm(A, B)
#            BA = torch.mm(B, A)
#            I = torch.eye(AB.shape[0]).float().cuda()
#            
#            loss_consist = (torch.sum((AB-I)**2)**0.5 + torch.sum((BA-I)**2)**0.5)
#            loss = loss +  0 * loss_consist            



            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()           

            


     

            


    
        # schedule learning rate decay    
        exp_lr_scheduler(optimizer, epoch, lr_decay_rate=learning_rate_change, decayEpoch=epoch_update)
        loss_hist.append(loss)                
        epoch_loss.append(epoch)
        
        
        if (epoch) % 20 == 0:
                print('********** Epoche %s **********' %(epoch+1))
                
                print("loss identity: ", loss_identity.item())
                print("loss prediction: ", loss_pred.item())
                print("loss sum: ", loss.item())

                epoch_hist.append(epoch+1) 

                w, _ = np.linalg.eig(model.dynamics.dynamics.weight.data.cpu().numpy())
                if gamma > 0:
                    #print(wP)
                    print(np.abs(w))

                #print(w)
    plt.figure()
    plt.plot(epoch_loss[1:], loss_hist[1:])
    plt.yscale('log')
    plt.savefig('loss' +'.png')
    plt.close()

                
                
    return model, optimizer, epoch_hist
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
import matplotlib.pyplot as plt
import torch.nn.init as init

import numpy as np

from tools import *

from Adam_new import *


def train_vae(model, train_loader, test_loader, lr, weight_decay,
          lamb, num_epochs, learning_rate_change, epoch_update, nu=0.0, eta=0.0, backward=0, steps=1, steps_back=1):
    
    
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
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

#    def wd_scheduler(optimizer, weight_decay):
#        """Decay learning rate by a factor of lr_decay_rate every lr_decay_epoch epochs"""
#        for param_group in optimizer.param_groups:
#            param_group['weight_decay_adapt'] = weight_decay
#        return optimizer

    #criterion = nn.MSELoss()#.cuda()


    epoch_hist = []
    loss_hist = []
    epoch_loss = []

    for epoch in range(num_epochs):

        for batch_idx, data_list in enumerate(train_loader):

            model.train()
            beta = 0.0001
            gamma = 0.0001

            data_list = [d.to(device) for d in data_list]    

            reconstruction_y_i, _, _, _, _, _ = model(data_list[0], mode='forward')
            
            mse, entropy, dynamic_mse, dynamic_entropy, loss_identity = \
                model.loss_function_multistep(data_list, reconstruction_y_i)
                
            loss_fwd = mse - gamma * entropy + beta * (dynamic_mse + dynamic_entropy) 


            loss_identity *= steps

            loss_bwd = 0.0 
            loss_consist = 0.0
            if backward == 1:
                _, _, _, _, _, reconstruction_y_i_back = model(data_list[-1], mode='backward')
                
                mse_b, entropy_b, dynamic_mse_b, dynamic_entropy_b, loss_identity_b = \
                    model.loss_function_multistep(list(reversed(data_list))[0:2], reconstruction_y_i_back, backward=True)

                loss_bwd = (mse_b - gamma * entropy_b + beta * (dynamic_mse_b + dynamic_entropy_b))
                #loss_identity += loss_identity_b
    
                # AB = I and BA = I
                A = model.dynamics.dynamics.weight
                B = model.backdynamics.dynamics.weight
                # AB = torch.mm(A, B)
                # BA = torch.mm(B, A)
                # I = torch.eye(AB.shape[0]).float().to(device)
                # loss_consist = eta * (torch.sum((AB-I)**2)**0.5 + torch.sum((BA-I)**2)**0.5)

                K = A.shape[-1]
                
                for k in range(1,K+1):
                    As1 = A[:,:k]
                    Bs1 = B[:k,:]
                    As2 = A[:k,:]
                    Bs2 = B[:,:k]

                    Ik = torch.eye(k).float().to(device)

                    if k == 1:
                        loss_consist = (torch.sum( (torch.mm(Bs1, As1)-Ik )**2)**1 + \
                                         torch.sum( (torch.mm(As2, Bs2)-Ik )**2)**1 ) / (2.0*k)
                    else:
                        loss_consist += (torch.sum( (torch.mm(Bs1, As1)-Ik )**2)**1 + \
                                         torch.sum( (torch.mm(As2, Bs2)-Ik )**2)**1 ) / (2.0*k)   
                
                
            loss = loss_fwd  + lamb * loss_identity + nu * loss_bwd + eta * loss_consist


            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #TODO: regularization wit gaussian prior once per epoch

        # schedule learning rate decay
        exp_lr_scheduler(optimizer, epoch, lr_decay_rate=learning_rate_change, decayEpoch=epoch_update)
        loss_hist.append(loss)
        epoch_loss.append(epoch)

        if (epoch) % 20 == 0:
            print('********** Epoch %s **********' % (epoch + 1))

            print("loss identity: ", loss_identity.item())
            if backward == 1:
                    print("loss backward: ", loss_bwd.item())
                    print("loss consistent: ", loss_consist.item())
            print("loss forward: ", loss_fwd.item())
            print("loss sum: ", loss.item())

            w, _ = np.linalg.eig(model.dynamics.dynamics.weight.data.cpu().numpy())
            print(np.abs(w))

    plt.figure()
    plt.plot(epoch_loss[1:], loss_hist[1:])
    plt.yscale('log')
    plt.savefig('loss' + '.png')
    plt.close()

    return model, optimizer, epoch_hist
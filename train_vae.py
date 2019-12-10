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
          lamb, num_epochs, learning_rate_change, epoch_update, gamma=0.0, backward=False):
    # optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)

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

    #criterion = nn.MSELoss()#.cuda()

    error_train = []
    error_test = []
    epoch_hist = []
    epoch_hist = []
    loss_hist = []
    epoch_loss = []

    for epoch in range(num_epochs):

        for batch_idx, data_list in enumerate(train_loader):

            model.train()
            beta = 0.0001
            gamma = 0.0001
            #for d in data_list: 
            #    d.to(device)
                
            data_list = [d.to(device) for d in data_list]    

            reconstruction_y_i, _, _, _, _, _ = model(data_list[0])
            mse, entropy, dynamic_mse, dynamic_entropy, loss_identity = \
                model.module.loss_function_multistep(data_list, reconstruction_y_i)
                
            loss = mse - gamma * entropy + beta * (dynamic_mse + dynamic_entropy) + lamb * loss_identity

            if backward:
                _, _, _, _, _, reconstruction_y_i_back = model(data_list[-1])
                mse_b, entropy_b, dynamic_mse_b, dynamic_entropy_b, loss_identity_b = \
                    model.module.loss_function_multistep(list(reversed(data_list)), reconstruction_y_i, backward=True)

                #loss += (mse_b - gamma * entropy_b + beta * (dynamic_mse_b + dynamic_entropy_b) + lamb * loss_identity_b) * 1e-7
                loss += (mse_b - gamma * entropy_b + beta * (dynamic_mse_b + dynamic_entropy_b)) * 1e-9

    
                # AB = I and BA = I
                A = model.module.dynamics.dynamics.weight
                B = model.module.backdynamics.dynamics.weight
                AB = torch.mm(A, B)
                BA = torch.mm(B, A)
                I = torch.eye(AB.shape[0]).float().to(device)
                
                loss_consist = 1e-9 * (torch.sum((AB-I)**2)**0.5 + torch.sum((BA-I)**2)**0.5)
                loss += loss_consist


            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #TODO: regularization wit gaussian prior once per epoch
            #TODO: confidence interval
            
            

        # schedule learning rate decay
        exp_lr_scheduler(optimizer, epoch, lr_decay_rate=learning_rate_change, decayEpoch=epoch_update)
        loss_hist.append(loss)
        epoch_loss.append(epoch)

        if (epoch) % 20 == 0:
            print('********** Epoch %s **********' % (epoch + 1))

            #print("loss identity: ", loss_identity.item())
            #print("loss prediction: ", loss_pred.item())
            print("loss sum: ", loss.item())

            error_train.append(error_summary(train_loader, model.eval(), 'train'))
            error_test.append(error_summary(test_loader, model.eval(), 'test'))
            epoch_hist.append(epoch + 1)

            w, _ = np.linalg.eig(model.module.dynamics.dynamics.weight.data.cpu().numpy())
            if gamma > 0:
                # print(wP)
                print(np.abs(w))

            # print(w)
    plt.figure()
    plt.plot(epoch_loss[1:], loss_hist[1:])
    plt.yscale('log')
    plt.savefig('loss' + '.png')
    plt.close()

    return model, optimizer, error_train, error_test, epoch_hist
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
          lamb, num_epochs, learning_rate_change, epoch_update, gamma=0.0):
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
            loss = 0
            beta = 0.0001
            gamma = 0.0001
            
            reconstruction_y_i, mu_i, logvar_i, dinamyc_ip, x_i = model(data_list[0].to(device))
            for k in range(len(data_list) - 1):
                _, mu_i, logvar_i, dinamyc_ip, x_i = model(data_list[k].to(device))
                _, mu_ip, logvar_ip, dinamyc_ipp, x_ip = model(data_list[k+1].to(device))
                mse, entropy, dynamic_mse, dynamic_entropy = model.module.loss_function(reconstruction_y_i[k], data_list[k+1].to(device),
                                                      mu_i, mu_ip, logvar_i, logvar_ip, first_step=False)
                loss += mse - gamma*entropy/data_list[0].shape[0] + beta*(dynamic_mse + dynamic_entropy)

            loss_identity = torch.nn.functional.mse_loss(reconstruction_y_i[len(data_list) - 1], data_list[0].to(device))
            loss = loss + lamb * loss_identity  # +  0 * loss_back
            #print(loss_identity)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #TODO: regularization only once per epoch
            #TODO: forward and backward dynamics
            #TODO: clean up code with .to(device)
            #TODO: confidence interval

            # schedule learning rate decay
        exp_lr_scheduler(optimizer, epoch, lr_decay_rate=learning_rate_change, decayEpoch=epoch_update)
        loss_hist.append(loss)
        epoch_loss.append(epoch)

        if (epoch) % 20 == 0:
            print('********** Epoche %s **********' % (epoch + 1))

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
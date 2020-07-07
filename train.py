import torch
from torch import nn
import numpy as np

from tools import *



def train(model, train_loader, lr, weight_decay, 
          lamb, num_epochs, learning_rate_change, epoch_update, 
          nu=0.0, eta=0.0, backward=0, steps=1, steps_back=1, gradclip=1):

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    device = get_device()
             
            
    def lr_scheduler(optimizer, epoch, lr_decay_rate=0.8, decayEpoch=[]):
                    """Decay learning rate by a factor of lr_decay_rate every lr_decay_epoch epochs"""
                    if epoch in decayEpoch:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= lr_decay_rate
                        return optimizer
                    else:
                        return optimizer
                        
                     
        

    criterion = nn.MSELoss().to(device)


    epoch_hist = []
    loss_hist = []
    epoch_loss = []
                            
    for epoch in range(num_epochs):
        #print(epoch)
        for batch_idx, data_list in enumerate(train_loader):
            model.train()
            out, out_back = model(data_list[0].to(device), mode='forward')


            for k in range(steps):
                if k == 0:
                    loss_fwd = criterion(out[k], data_list[k+1].to(device))
                else:
                    loss_fwd += criterion(out[k], data_list[k+1].to(device))

            
            loss_identity = criterion(out[-1], data_list[0].to(device)) * steps


            loss_bwd = 0.0
            loss_consist = 0.0

            loss_bwd = 0.0
            loss_consist = 0.0

            if backward == 1:
                out, out_back = model(data_list[-1].to(device), mode='backward')
   

                for k in range(steps_back):
                    
                    if k == 0:
                        loss_bwd = criterion(out_back[k], data_list[::-1][k+1].to(device))
                    else:
                        loss_bwd += criterion(out_back[k], data_list[::-1][k+1].to(device))
                        
                               
                A = model.dynamics.dynamics.weight
                B = model.backdynamics.dynamics.weight

                K = A.shape[-1]

                for k in range(1,K+1):
                    As1 = A[:,:k]
                    Bs1 = B[:k,:]
                    As2 = A[:k,:]
                    Bs2 = B[:,:k]

                    Ik = torch.eye(k).float().to(device)

                    if k == 1:
                        loss_consist = (torch.sum((torch.mm(Bs1, As1) - Ik)**2) + \
                                         torch.sum((torch.mm(As2, Bs2) - Ik)**2) ) / (2.0*k)
                    else:
                        loss_consist += (torch.sum((torch.mm(Bs1, As1) - Ik)**2) + \
                                         torch.sum((torch.mm(As2, Bs2)-  Ik)**2) ) / (2.0*k)


                
                
                
#                Ik = torch.eye(K).float().to(device)
#                loss_consist = (torch.sum( (torch.mm(A, B)-Ik )**2)**1 + \
#                                         torch.sum( (torch.mm(B, A)-Ik)**2)**1 )
#   
                                        
                
    
            loss = loss_fwd + lamb * loss_identity +  nu * loss_bwd + eta * loss_consist

            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradclip) # gradient clip
            optimizer.step()           

        # schedule learning rate decay    
        lr_scheduler(optimizer, epoch, lr_decay_rate=learning_rate_change, decayEpoch=epoch_update)
        loss_hist.append(loss)                
        epoch_loss.append(epoch)
        
        
        if (epoch) % 20 == 0:
                print('********** Epoche %s **********' %(epoch+1))
                
                print("loss identity: ", loss_identity.item())
                if backward == 1:
                    print("loss backward: ", loss_bwd.item())
                    print("loss consistent: ", loss_consist.item())
                print("loss forward: ", loss_fwd.item())
                print("loss sum: ", loss.item())

                epoch_hist.append(epoch+1) 

                if hasattr(model.dynamics, 'dynamics'):
                    w, _ = np.linalg.eig(model.dynamics.dynamics.weight.data.cpu().numpy())
                    print(np.abs(w))


    if backward == 1:
        loss_consist = loss_consist.item()
                
                
    return model, optimizer, [epoch_hist, loss_fwd.item(), loss_consist]

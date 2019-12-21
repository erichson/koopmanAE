from torch import nn
from torch.autograd import grad
import torch
from torch.autograd import Variable, Function

ALPHA = 2

class FullResBlock(torch.nn.Module):
    def __init__(self, in_features, out_features, act=torch.nn.functional.tanh):
        super(FullResBlock,self).__init__()
        self.act = act
        
        self.L1 = nn.Linear(in_features, int(in_features/2))
        self.bn1 = torch.nn.functional.batch_norm       
        self.L2 = nn.Linear(int(in_features/2), out_features)
        self.bn2 = torch.nn.functional.batch_norm
        
    def forward(self, x):
        residum = x
        x = self.L1(x)
        x = self.act(x)
        x = self.bn1(x)        
        x = self.L2(x)
        x = self.act(x)
        x = self.bn2(x) 
        return x+residum


class encoderNet(nn.Module):
    def __init__(self, m, n, b):
        super(encoderNet, self).__init__()
        self.N = m * n
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(self.N, 32*ALPHA)
        self.bn = nn.BatchNorm1d(1)
        
        self.resblock1 = FullResBlock(32*ALPHA, 32*ALPHA)
        self.resblock2 = FullResBlock(32*ALPHA, 32*ALPHA)
        self.fc4 = nn.Linear(32*ALPHA, b)
        

    def forward(self, x):
        x = x.view(-1, 1, self.N)
        x = self.tanh(self.fc1(x))
        #x = self.bn(x)
        x = self.resblock1(x)     
        x = self.resblock2(x)
        x = self.fc4(x)
        return x


class decoderNet(nn.Module):
    def __init__(self, m, n, b):
        super(decoderNet, self).__init__()

        self.m = m
        self.n = n
        self.b = b

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(b, 32*ALPHA)
        self.bn = nn.BatchNorm1d(1)        
        self.resblock1 = FullResBlock(32*ALPHA, 32*ALPHA)
        self.resblock2 = FullResBlock(32*ALPHA, 32*ALPHA)
        self.fc4 = nn.Linear(32*ALPHA, m*n)




    def forward(self, x):
        x = x.view(-1, 1, self.b)
        x = self.tanh(self.fc1(x))
        #x = self.bn(x)
        x = self.resblock1(x)     
        x = self.resblock2(x)
        x = self.fc4(x)
        x = x.view(-1, 1, self.m, self.n)
        return x





class dynamics(nn.Module):
    def __init__(self, b):
        super(dynamics, self).__init__()
        self.dynamics = nn.Linear(b, b, bias=False)

    def forward(self, x):
        # 1 to 1 map
        x = self.dynamics(x)
        return x


class backdynamics(nn.Module):
    def __init__(self, b):
        super(backdynamics, self).__init__()
        self.dynamics = nn.Linear(b, b, bias=False)      

    def forward(self, x):
        # 1 to 1 map      
        x = self.dynamics(x)
        return x      
    


class shallow_autoencoder(nn.Module):
    def __init__(self, m, n, b, steps):
        super(shallow_autoencoder, self).__init__()
        self.steps = steps
        self.encoder = encoderNet(m, n, b)
        self.dynamics = dynamics(b)
        #self.backdynamics = backdynamics(b)
        self.decoder = decoderNet(m, n, b)

    def forward(self, x):
        out = []
        out_back = []
        z = self.encoder(x.contiguous())


        q = z.contiguous()
        
        for _ in range(self.steps):
            #q = q + self.dynamics(q)
            #out.append(self.decoder(q))
            
            q = self.dynamics(q)
            out.append(self.decoder(q.contiguous()))

            #q_back = self.backdynamics(q.contiguous())
            #out_back.append(self.decoder(q_back))
    

        out.append(self.decoder(z.contiguous())) # Identity


        return out, out_back

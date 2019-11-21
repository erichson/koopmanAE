from torch import nn
from torch.autograd import grad
import torch
from torch.autograd import Variable, Function


class encoderNet(nn.Module):
    def __init__(self, m, n, b):
        super(encoderNet, self).__init__()
        self.N = m * n
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(self.N, 32)
        self.fc2 = nn.Linear(32, 8)
        self.fc3 = nn.Linear(8, b)

    def forward(self, x):
        x = x.view(-1, 1, self.N)
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.tanh(self.fc3(x))
        #x = self.bn(x)
        return x


class decoderNet(nn.Module):
    def __init__(self, m, n, b):
        super(decoderNet, self).__init__()

        self.m = m
        self.n = n

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(b, 8)
        self.fc2 = nn.Linear(8, 32)
        self.fc3 = nn.Linear(32, m*n)


    def forward(self, x):
        #x = self.tanh(x)
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)
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

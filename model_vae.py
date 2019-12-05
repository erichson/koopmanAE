from torch import nn
from torch.autograd import grad
import torch
from torch.autograd import Variable, Function
import math


class VEncoderNet(nn.Module):
    def __init__(self, m, n, b):
        super(VEncoderNet, self).__init__()
        self.N = m * n
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(self.N, 32)
        self.fc2 = nn.Linear(32, 8)
        self.fc3mu = nn.Linear(8, b)
        self.fc3logvar = nn.Linear(8, b)

    def forward(self, x):
        x = x.view(-1, 1, self.N)
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        mu = self.tanh(self.fc3mu(x))
        logvar = self.fc3logvar(x)

        return mu, logvar


class VDecoderNet(nn.Module):
    def __init__(self, m, n, b):
        super(VDecoderNet, self).__init__()

        self.m = m
        self.n = n

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(b, 8)
        self.fc2 = nn.Linear(8, 32)
        self.fc3 = nn.Linear(32, m * n)

    def forward(self, x):
        # x = self.tanh(x)
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, 1, self.m, self.n)
        return x


class Dynamics(nn.Module):
    def __init__(self, b):
        super(Dynamics, self).__init__()
        self.dynamics = nn.Linear(b, b, bias=False)

    def forward(self, x):
        # 1 to 1 map
        x = self.dynamics(x)
        return x

    def forward_dist(self, mu, cov, step):
        mu_p = self.dynamics(mu)
        omega = self.dynamics.weight
        cov = torch.squeeze(cov)
        if len(cov.shape) < 3:
            cov = torch.diag_embed(cov)
        cholesky_cov = torch.matmul(torch.matrix_power(omega, step), torch.sqrt(cov))
        return mu_p, cholesky_cov


class BackDynamics(nn.Module):
    def __init__(self, b):
        super(BackDynamics, self).__init__()
        self.dynamics = nn.Linear(b, b, bias=False)

    def forward(self, x):
        # 1 to 1 map
        x = self.dynamics(x)
        return x


class DynamicVAE(nn.Module):
    def __init__(self, m, n, b, steps):
        super(DynamicVAE, self).__init__()
        self.steps = steps
        self.encoder = VEncoderNet(m, n, b)
        self.dynamics = Dynamics(b)
        self.decoder = VDecoderNet(m, n, b)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def reparametrize_multidim(self, mu, cholesky_cov):
        mu = torch.squeeze(mu)
        eps = torch.randn_like(mu)
        eps = eps.unsqueeze(-1)
        return mu + torch.bmm(cholesky_cov, eps).squeeze()

    def forward(self, y_i):
        out = []
        mu_i, logvar_i = self.encoder.forward(y_i)
        x_i = self.reparametrize(mu_i, logvar_i)
        dynamic_ip = self.dynamics.forward(x_i)
        q_mu = mu_i.contiguous()
        q_logvar = logvar_i.contiguous()
        q_var_i = torch.exp(0.5*q_logvar)
        for i in range(self.steps):
            q_mu, q_var = self.dynamics.forward_dist(q_mu, q_var_i, i+1)
            x = self.reparametrize_multidim(q_mu, q_var)
            out.append(self.decoder.forward(x))
        out.append(self.decoder.forward(x_i))
        return out, mu_i, logvar_i, dynamic_ip, x_i

    def loss_function(self, reconstruction_y_i, y_i, mu_i, mu_ip, logvar_i, logvar_ip, first_step=False):
        mse = torch.nn.functional.mse_loss(reconstruction_y_i, y_i)
        entropy = 0.5 * torch.mean(logvar_i)
        dynamic_mse = torch.nn.functional.mse_loss(mu_ip, self.dynamics.forward(mu_i))
        dynamic_entropy = torch.mean(logvar_ip.exp()) + \
                          torch.mean(logvar_i.exp() * torch.diag(self.dynamics.dynamics.weight).pow(2))
        if first_step:
            logvar_ip = logvar_i
            mu_ip = mu_i
            mu_i = torch.zeros_like(mu_ip)
            logvar_i = torch.zeros_like(logvar_ip)
            dynamic_loss = torch.nn.functional.mse_loss(mu_ip, self.dynamics.forward(mu_i)) + \
                           torch.mean(torch.exp(logvar_ip)) + \
                           torch.mean(torch.exp(logvar_i) * torch.diag(self.dynamics.dynamics.weight).pow(2))
        return mse, entropy, dynamic_mse, dynamic_entropy

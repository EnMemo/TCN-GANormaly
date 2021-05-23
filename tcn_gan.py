import torch
import torch.nn as nn
from .tools import device, initNetParams
from .tcn import TemporalConvNet

def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + eps*std

class Encoder(nn.Module):
    def __init__(self, dim_x, dim_z, cons):
        super(Encoder, self).__init__()
        self.tcn = TemporalConvNet(dim_x, cons)
        self.tcn_out_1 = TemporalConvNet(cons[-1], [dim_z])
        self.tcn_out_2 = TemporalConvNet(cons[-1], [dim_z])

    def forward(self, x):
        return self.tcn_out_1(self.tcn(x)), self.tcn_out_2(self.tcn(x))

class Decoder(nn.Module):
    def __init__(self, dim_z, dim_x, cons):
        super(Decoder, self).__init__()
        cons.reverse()
        self.tcn_out = TemporalConvNet(dim_z, cons)
        self.tcn = TemporalConvNet(cons[-1], [dim_x])
        cons.reverse()

    def forward(self, z):
        return self.tcn(self.tcn_out(z))

class Discriminator(nn.Module):
    def __init__(self, dim_x, cons):
        super(Discriminator, self).__init__()
        self.tcn = TemporalConvNet(dim_x, cons)
        self.tcn_out = TemporalConvNet(cons[-1], [1])

    def forward(self, x):
        mid_output = self.tcn(x)
        return self.tcn_out(mid_output), mid_output

class VAE(nn.Module):
    def __init__(self, dim_x, dim_z, cons):
        super(VAE, self).__init__()
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.cons = cons
        self.generator_E = Encoder(self.dim_x, self.dim_z, self.cons)
        self.generator_D = Decoder(self.dim_z, self.dim_x, self.cons)
    
    def forward(self, x):
        z_avg, z_log_var = self.generator_E(x)
        z = reparameterize(z_avg, z_log_var)
        x_rec = self.generator_D(z)
        return z, x_rec, z_avg, z_log_var 

class Generator(nn.Module):
    def __init__(self, dim_x, dim_z, cons):
        super(Generator, self).__init__()
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.cons = cons
        self.VAE = VAE(self.dim_x, self.dim_z, self.cons)
        self.encoder = Encoder(self.dim_x, self.dim_z, self.cons)

    def forward(self, x):
        x = x.permute(0,2,1) 
        z, x_rec, z_avg, z_log_var = self.VAE(x)
        z_avg_rec, z_log_var_rec = self.encoder(x_rec)
        z_rec = reparameterize(z_avg_rec, z_log_var_rec)
        return z, x_rec, z_rec, z_avg, z_log_var    


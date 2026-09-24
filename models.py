import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from configs import *
from components import ActivationFunction, CKConv1d, FiLMPCINR
   
class LatentMappingNetwork(nn.Module):
    """
    Maps the local latent triplet + relative time to a SINGLE global FiLM 
    scaling (gamma) and shifting (beta) parameter pair applied to all layers.
    Mathematically mirrors the official PiGANMappingNetwork.
    """
    def __init__(self, in_features, hidden_dim=mdim):
        super().__init__()
        self.hidden_dim = hidden_dim
        out_features = hidden_dim * 2
        
        layers = []
        layers.extend([nn.Linear(in_features, hidden_dim), nn.LeakyReLU(0.2, inplace=True)])
        nn.init.kaiming_normal_(layers[-2].weight, mode='fan_in', nonlinearity='leaky_relu', a=0.2)
        
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2, inplace=True)])
        nn.init.kaiming_normal_(layers[-2].weight, mode='fan_in', nonlinearity='leaky_relu', a=0.2)
        
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2, inplace=True)])
        nn.init.kaiming_normal_(layers[-2].weight, mode='fan_in', nonlinearity='leaky_relu', a=0.2)
        
        layers.extend([nn.Linear(hidden_dim, out_features)])
        nn.init.kaiming_normal_(layers[-1].weight, mode='fan_in', nonlinearity='leaky_relu', a=0.2)
        
        self.net = nn.Sequential(*layers)
        with torch.no_grad():
            self.net[-1].weight *= 0.25

    def forward(self, x):
        out = self.net(x)
        
        gammas, betas = out[..., :self.hidden_dim], out[..., self.hidden_dim:]
        gammas = 30.0 * gammas / (1.0 + torch.abs(gammas))
        betas = math.pi * betas / (1.0 + torch.abs(betas))
        
        return gammas, betas

class DynamicEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = CKConv1d(1, 16, time_window_ms=7/13*ckconv_window)
        self.act1 = ActivationFunction()
        
        self.conv2 = CKConv1d(16, 32, time_window_ms=3/13*ckconv_window)
        self.act2 = ActivationFunction()
        
        self.conv3 = CKConv1d(32, 64, time_window_ms=3/13*ckconv_window)
        self.act3 = ActivationFunction()
        
        self.conv_out = CKConv1d(64, encoder_dim, time_window_ms=0.0)

    def forward(self, x, sr):
        x = self.act1(self.conv1(x, sr))
        x = self.act2(self.conv2(x, sr))
        x = self.act3(self.conv3(x, sr))
        z = self.conv_out(x, sr)
        return z
    
class SIRIUS(nn.Module):
    def __init__(self, mean=0.0, std=0.0594):
        super().__init__()
        self.register_buffer('data_mean', torch.tensor(mean))
        self.register_buffer('data_std', torch.tensor(std))
        
        self.encoder = DynamicEncoder()
        
        in_features = (3 * encoder_dim)
        self.num_inr_layers = 4
        
        self.mapping_net = LatentMappingNetwork(in_features, hidden_dim=mdim)
        self.pcinr = FiLMPCINR(num_layers=self.num_inr_layers, hidden_dim=mdim)

    def forward(self, x:torch.Tensor, low_sr, high_sr):
        scale = high_sr / low_sr
        x_norm = (x - self.data_mean) / (self.data_std + 1e-8)
        B, _, L_lr = x_norm.shape

        z = self.encoder(x_norm, low_sr)
        if self.training: z += torch.randn_like(z) * 0.001
        z_pad = F.pad(z, (1, 1), mode='replicate')
        z_prev = z_pad[:, :, :-2]
        z_curr = z_pad[:, :, 1:-1]
        z_next = z_pad[:, :, 2:]
        
        z_triplet = torch.cat([z_prev, z_curr, z_next], dim=1).transpose(1, 2)
        gammas, betas = self.mapping_net(z_triplet)

        L_hr = int(L_lr * scale)
        t_mapped = torch.arange(L_hr, device=z.device).float() / scale
        t_mapped = t_mapped.unsqueeze(0).repeat(B, 1)
        
        if self.training:
            eta = torch.randn_like(t_mapped) * (0.5 / scale)
            t_select = t_mapped + eta
        else:
            t_select = t_mapped

        idx_i = torch.floor(t_select + 0.5).long().clamp(0, L_lr - 1)
        t_rel = (t_mapped - idx_i.float()).unsqueeze(-1) 
        
        expanded_idx = idx_i.unsqueeze(-1).expand(-1, -1, gammas.size(2))
        gammas = torch.gather(gammas, 1, expanded_idx)
        betas = torch.gather(betas, 1, expanded_idx)
        out = self.pcinr(t_rel, gammas, betas)
        out = out.transpose(1, 2).contiguous()
        out = (out * self.data_std) + self.data_mean
            
        return out
        
    def load_checkpoint(self, path, device):
        """Standardized checkpoint loader for evaluation wrappers."""
        checkpoint = torch.load(path, map_location=device)
        
        # Prioritize EMA weights if they exist, fallback to standard model weights
        state_key = 'ema' if 'ema' in checkpoint else 'model'
        
        self.load_state_dict(checkpoint[state_key], strict=True)
import torch
import torch.nn as nn
import torch.nn.functional as F
from extraUtils.layer import getActivation, Swiglu
from configs import *
import math

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.trunc = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            getActivation(actfe),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            getActivation(actfe),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            getActivation(actfe))
        
        self.mean_head = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            getActivation(actfe),
            nn.Conv1d(64, encoder_dim, kernel_size=1)
        )
        self.std_head = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            getActivation(actfe),
            nn.Conv1d(64, encoder_dim, kernel_size=1)
        )
        
    def forward(self, x):
        x = self.trunc(x)
        mean = self.mean_head(x)
        std = torch.exp(0.5*(self.std_head(x).clamp(min=-8,max=8)))
        return mean, std
    
class ConvolutionalDecoder(nn.Module):
    def __init__(self, mean=0.0, std=1):
        super().__init__()
        
        self.register_buffer('mean_data', torch.tensor(mean))
        self.register_buffer('std_data', torch.tensor(std))
        
        self.trunc = nn.Sequential(
            nn.ConvTranspose1d(in_channels=encoder_dim, out_channels=64,kernel_size=1, padding=0),
            getActivation(actfdc),
            nn.ConvTranspose1d(in_channels=64, out_channels=64,kernel_size=3, padding=1),
            getActivation(actfdc),
            nn.ConvTranspose1d(in_channels=64, out_channels=32,kernel_size=3, padding=1),
            getActivation(actfdc),
            nn.ConvTranspose1d(in_channels=32, out_channels=16,kernel_size=3, padding=1),
            getActivation(actfdc),
            nn.ConvTranspose1d(in_channels=16, out_channels=1,kernel_size=7, padding=3)
        )
        
    def forward(self, x):
        return self.trunc(x) * self.std_data + self.mean_data
    
class DecoderBlock(nn.Module):
    
    def __init__(self, in_features, out_features,*args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.act1 = nn.RMSNorm(in_features)
        self.act2 = Swiglu(in_features=in_features,out_features=out_features)
        self.proj = nn.Linear(in_features=out_features,out_features=out_features)
        
    def forward(self, x):
        
        skip = x
        y = self.proj(self.act2(self.act1(x)))
        return skip + y
     
class INRDecoder(nn.Module):
    
    def __init__(self, mean=0.0, std=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.register_buffer('mean_data', torch.tensor(mean))
        self.register_buffer('std_data', torch.tensor(std))
        
        self.decoder = nn.Sequential(
            nn.Linear(freq_bands * 2 + 2 + encoder_dim * 3, 196),
            nn.SiLU(),
            nn.Linear(196, 196),
            DecoderBlock(196, 196),
            DecoderBlock(196, 196),
            DecoderBlock(196, 196),
            nn.SiLU(),
            nn.Linear(196, 1)
        )
        
    def forward(self, z:torch.Tensor, scale):
        
        B, _, L_lr = z.shape
        L_hr = int(L_lr * scale)
        t_hr = torch.arange(L_hr, device=z.device).float() / scale
        t_hr = t_hr.unsqueeze(0).repeat(B, 1)
        
        if self.training:
            eta = torch.randn_like(t_hr) * 0.5
            t_select = t_hr + eta
        else:
            t_select = t_hr
        
        idx_i = torch.round(t_select).long().clamp(0, L_lr - 1)
        t_i = idx_i.float()
        
        t_rel = (t_hr - t_i).unsqueeze(-1)
        z_pad = F.pad(z, (1, 1), mode='replicate')
        idx_curr = idx_i + 1
        idx_prev = idx_i
        idx_next = idx_i + 2
        
        def collect(idx:torch.Tensor):
            expanded_idx = idx.unsqueeze(1).expand(-1, z.shape[1], -1)
            return torch.gather(z_pad, 2, expanded_idx)

        z_triplet = torch.cat([collect(idx_prev), collect(idx_curr), collect(idx_next)], dim=1)
        z_triplet = z_triplet.transpose(1, 2)
        
        freqs = 2.0 ** torch.arange(freq_bands, device=z.device)
        scaled_t = t_rel * freqs * math.pi
        sin_t = torch.sin(scaled_t)
        cos_t = torch.cos(scaled_t)
        t_pe = torch.stack([sin_t, cos_t], dim=-1).flatten(start_dim=-2)
        t_encoded = torch.cat([t_rel, t_pe], dim=-1)
        scale_tensor = torch.full_like(t_rel, math.log(scale))
        
        feat = torch.cat([t_encoded, z_triplet, scale_tensor], dim=-1)
        out = self.decoder(feat)
        
        return out.transpose(1, 2).contiguous() * self.std_data + self.mean_data
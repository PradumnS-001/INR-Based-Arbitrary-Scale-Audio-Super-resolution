import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import *
from extraUtils.activations import getActivation
from extraUtils.loss import gamma_loss

class AffineTransformation(nn.Module):
    
    def __init__(self, in_feats = 128, out_feats = 128):
        super().__init__()
        
        self.fc = nn.Linear(in_feats, out_feats)
        self.A = nn.Linear(128,out_feats)
        self.B = nn.Linear(128,out_feats)
        self.C = nn.Sequential(
            nn.Linear(128,out_feats),
            nn.Sigmoid()
        )
        
    def forward(self,
                pre_alpha:torch.Tensor,
                pre_beta:torch.Tensor,
                pre_gamma:torch.Tensor, 
                canvas:torch.Tensor):
        
        alpha = self.A(pre_alpha)
        beta = self.B(pre_beta)
        gamma = self.C(pre_gamma)
        canvas = self.fc(canvas)
        noise = torch.randn_like(gamma)
        canvas = torch.sqrt(1-gamma) * canvas + torch.sqrt(gamma) * noise
        return alpha * canvas + beta, gamma_loss(gamma)
    
class StyleDecoder(nn.Module):
    
    def __init__(self, in_dim, out_dim):
        super().__init__()
        
        self.fc1 = AffineTransformation(in_dim,128)
        self.act1 = getActivation(act='gelu')
        self.fc2 = AffineTransformation(128,128)
        self.act2 = getActivation(act='gelu')
        self.fc3 = AffineTransformation(128,128)
        self.act3 = getActivation(act='gelu')
        self.fc4 = AffineTransformation(128,128)
        self.act4 = getActivation(act='gelu')
        self.fc5 = nn.Linear(128,out_dim)
        
    def forward(self,
                pre_alpha:torch.Tensor,
                pre_beta:torch.Tensor,
                pre_gamma:torch.Tensor, 
                canvas:torch.Tensor):
        
        o1,gl1 = self.fc1(pre_alpha, pre_beta, pre_gamma, canvas)
        o1 = self.act1(o1)
        o1,gl2 = self.fc1(pre_alpha, pre_beta, pre_gamma, canvas)
        o1 = self.act1(o1)
        o1,gl3 = self.fc1(pre_alpha, pre_beta, pre_gamma, canvas)
        o1 = self.act1(o1)
        o1,gl4 = self.fc1(pre_alpha, pre_beta, pre_gamma, canvas)
        o1 = self.act1(o1)
        o1 = self.fc5(o1)
        return o1, gl1+gl2+gl3+gl4
    
class ImprovedLISA(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            getActivation(act=actfe),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            getActivation(act=actfe),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            getActivation(act=actfe),
            nn.Conv1d(64, 32, kernel_size=1)
        )
        
        self.decoder = StyleDecoder(in_dim=num_bands*2+1,out_dim=1)
        
        self.style_mlp = nn.Sequential(
            nn.Linear(32*3+1,128),
            getActivation(act=actfs),
            nn.Linear(128,128),
            getActivation(act=actfs),
            nn.Linear(128,128),
            getActivation(act='gelu')
        )
        self.fch1 = nn.Linear(128,128)
        self.fch2 = nn.Linear(128,128)
        self.fch3 = nn.Linear(128,128)

    def forward(self, x_lr:torch.Tensor, scale:int | float): 
        
        B, _, L_lr = x_lr.shape
        L_hr = int(L_lr * scale)
        
        z = self.encoder(x_lr)
        
        t_hr = torch.arange(L_hr, device=x_lr.device).float() / scale
        t_hr = t_hr.unsqueeze(0).repeat(B, 1)
        
        if self.training:
            eta = torch.randn_like(t_hr) * 0.4
            t_select = t_hr + eta
        else:
            t_select = t_hr
        
        idx_i = torch.round(t_select).long().clamp(0, L_lr - 1)
        t_rel = (t_hr - idx_i.float()).unsqueeze(-1)
        
        freq_exps = torch.arange(num_bands, device=x_lr.device, dtype=torch.float32)
        frequencies = omega * (2.0 ** freq_exps)
        frequencies = frequencies.view(1, 1, -1)
        t_scaled = t_rel * frequencies
        pe = torch.cat([torch.sin(t_scaled), torch.cos(t_scaled), t_rel], dim=-1)
        
        z_pad = F.pad(z, (1, 1), mode='replicate')
        z_prev_base = z_pad[:, :, :-2]
        z_curr_base = z_pad[:, :, 1:-1]
        z_next_base = z_pad[:, :, 2:]
        z_triplet_base = torch.cat([z_prev_base, z_curr_base, z_next_base], dim=1).transpose(1, 2)

        scale_tensor = torch.full((B, L_hr, 1), float(scale), device=x_lr.device)
        feat = torch.cat([scale_tensor, z_triplet_base], dim=-1)
        
        base = self.style_mlp(feat)
        pre_alpha_base = self.fch1(base)
        pre_beta_base = self.fch2(base)
        pre_gamma_base = self.fch3(base)
        idx_expanded = idx_i.unsqueeze(-1).expand(-1, -1, 128)
        
        pre_alpha = torch.gather(pre_alpha_base, 1, idx_expanded)
        pre_beta  = torch.gather(pre_beta_base, 1, idx_expanded)
        pre_gamma = torch.gather(pre_gamma_base, 1, idx_expanded)
        
        out,gloss = self.decoder(pre_alpha,pre_beta,pre_gamma,pe)
        
        return out.mT.contiguous(), gloss
import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import *
from extraUtils.activations import getActivation
from extraUtils.loss import gamma_loss

class AffineTransformation(nn.Module):
    
    def __init__(self, in_feats = mdim, out_feats = mdim):
        super().__init__()
        
        self.fc = nn.Linear(in_feats, out_feats, bias=False)
        self.actv = getActivation(actfd)
        
    def forward(self,
                alpha:torch.Tensor,
                beta:torch.Tensor,
                gamma:torch.Tensor, 
                canvas:torch.Tensor):
        
        x = self.fc(canvas * alpha)
        weight_sq = self.fc.weight.pow(2)
        alpha_sq = alpha.pow(2)
        demod = torch.rsqrt(F.linear(alpha_sq, weight_sq) + 1e-8)
        canvas = x * demod
        
        noise = torch.randn_like(canvas)
        canvas += gamma**2 * noise
        return self.actv(canvas + beta)
    
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
        
        self.style_mlp = nn.Sequential(
            nn.Linear(32*3+1,mdim),
            getActivation(act=actfs),
            nn.Linear(mdim,mdim),
            getActivation(act=actfs),
            nn.Linear(mdim,mdim),
            getActivation(act=actfs)
        )
        self.fch1 = nn.Linear(mdim,mdim)
        self.fch2 = nn.Linear(mdim,mdim)
        self.fch3 = nn.Linear(mdim,mdim)
        
        self.alpha_transforms = nn.ModuleList([nn.Linear(mdim, num_bands*2+1)] + [nn.Linear(mdim, mdim) for _ in range(3)])
        self.beta_transforms = nn.ModuleList([nn.Linear(mdim,mdim) for _ in range(4)])
        self.gamma_transforms = nn.ModuleList([nn.Sequential(nn.Linear(mdim,mdim),nn.Sigmoid()) for _ in range(4)])
        self.affine_transformations = nn.ModuleList([AffineTransformation(num_bands*2+1, mdim)] + [AffineTransformation(mdim, mdim) for _ in range(3)])
        
        self.output = nn.Linear(mdim, 1)
        for m in self.gamma_transforms:
            nn.init.constant_(m[0].bias, -3.0)
        for m in self.alpha_transforms:
            nn.init.ones_(m.bias)

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

        scale_tensor = torch.ones(B,z_triplet_base.shape[1],1, device=x_lr.device,dtype=torch.float32) * torch.tensor(scale, device=x_lr.device,dtype=torch.float32)
        feat = torch.cat([scale_tensor, z_triplet_base], dim=-1)
        
        base = self.style_mlp(feat)
        pre_alpha_base = self.fch1(base)
        pre_beta_base = self.fch2(base)
        pre_gamma_base = self.fch3(base)
        
        alphas = [self.alpha_transforms[i](pre_alpha_base) for i in range(4)]
        betas = [self.beta_transforms[i](pre_beta_base) for i in range(4)]
        gammas = [self.gamma_transforms[i](pre_gamma_base) for i in range(4)]
        
        idx_expanded = idx_i.unsqueeze(-1).expand(-1, -1, mdim)
        
        gloss = 0
        for i in range(4):
            gloss += gamma_loss(gammas[i])*0.25
            alpha = torch.gather(alphas[i], 1, idx_expanded[:,:,:(alphas[i].shape)[-1]])
            beta = torch.gather(betas[i], 1, idx_expanded)
            gamma = torch.gather(gammas[i], 1, idx_expanded)
            pe = self.affine_transformations[i](alpha,beta,gamma,pe)
        
        return self.output(pe).mT.contiguous(), gloss
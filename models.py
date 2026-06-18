import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import *
from extraUtils.layers import getActivation, WeightNormLinear
from encodec.modules import SEANetEncoder

class AffineTransformation(nn.Module):
    
    def __init__(self, in_feats = mdim, out_feats = mdim, noisify:bool=True):
        super().__init__()
        
        self.norm = nn.RMSNorm(in_feats)
        self.fc1 = nn.Linear(in_feats, out_feats)
        self.fc2 = nn.Linear(out_feats,out_feats)
        self.actv = getActivation(actfd)
        if noisify:
            self.gamma_vec = nn.Parameter(torch.randn(1,1,out_feats) * temperature)
        self.noisify = noisify
        
    def forward(self,
                alpha:torch.Tensor,
                beta:torch.Tensor,
                gamma:torch.Tensor, 
                canvas:torch.Tensor,
                stochastic:bool = True):
        
        skip = canvas
        B, L, _ = canvas.shape
        canvas = self.fc1(self.norm(canvas) * alpha)
        weight_sq = self.fc1.weight.pow(2).float()
        alpha_sq = alpha.pow(2).float()
        demod = torch.rsqrt(F.linear(alpha_sq, weight_sq) + 1e-5)
        canvas = canvas * demod
        
        if self.noisify and stochastic:
            noise = torch.randn(B,L,1, device=canvas.device)
            canvas = canvas + gamma * self.gamma_vec * noise
        canvas = self.fc2(self.actv(canvas + beta))
        
        return skip + canvas
    
class ImprovedLISA(nn.Module):
    def __init__(self, opcs:torch.Tensor=torch.tensor(0.03)):
        super().__init__()
        
        self.macro_encoder = SEANetEncoder(n_filters=32, dimension=mdim, ratios=[4,4], lstm=0)
        self.macro_proj = nn.Conv1d(mdim, int(0.75*mdim), kernel_size=1)
        self.register_buffer('opcs', opcs.clamp_(min=0.001))
        
        self.micro_encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            getActivation(act=actfe),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            getActivation(act=actfe),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            getActivation(act=actfe),
            nn.Conv1d(64, mdim - int(0.75*mdim), kernel_size=1)
        )
        
        self.alpha_branch = nn.Sequential(
            WeightNormLinear(mdim * 3 + 1, mdim),
            getActivation(act=actfs),
            WeightNormLinear(mdim, mdim),
            getActivation(act=actfs),
            WeightNormLinear(mdim, mdim),
            getActivation(act=actfs),
            WeightNormLinear(mdim, mdim),
            getActivation(act=actfs),
            nn.Linear(mdim, num_blocks * mdim)
        )
        
        self.beta_branch = nn.Sequential(
            WeightNormLinear(mdim * 3 + 1, mdim),
            getActivation(act=actfs),
            WeightNormLinear(mdim, mdim),
            getActivation(act=actfs),
            WeightNormLinear(mdim, mdim),
            getActivation(act=actfs),
            WeightNormLinear(mdim, mdim),
            getActivation(act=actfs),
            nn.Linear(mdim, num_blocks * mdim)
        )
        
        self.gamma_branch = nn.Sequential(
            WeightNormLinear(mdim * 3 + 1, mdim),
            getActivation(act=actfs),
            WeightNormLinear(mdim, mdim),
            getActivation(act=actfs),
            WeightNormLinear(mdim, mdim),
            getActivation(act=actfs),
            WeightNormLinear(mdim, mdim),
            getActivation(act=actfs),
            nn.Linear(mdim, num_blocks * 1),
            nn.Sigmoid()
        )
        
        k = (num_bands+1)*2 + mdim
        self.input_projection = nn.Sequential(
            nn.Linear((num_bands+1)*2, k//4),
            getActivation(actfd),
            nn.Linear(k//4,k//2),
            getActivation(actfd),
            nn.Linear(k//2,mdim)
        )
        self.affine_transformations = nn.ModuleList([
            AffineTransformation(mdim, mdim, noisify=(i >= noisy_start))
            for i in range(num_blocks)
        ])
        
        self.output_block = nn.Sequential(
            WeightNormLinear(mdim, mdim // 2),
            getActivation(actfd),
            WeightNormLinear(mdim // 2, mdim // 2),
            getActivation(actfd),
            nn.Linear(mdim // 2, 1)
        )
        self.omega = nn.Parameter(torch.tensor(omega)) if is_omega_trainable else torch.tensor(omega)
        nn.init.constant_(self.alpha_branch[-1].bias, 1)
        nn.init.constant_(self.beta_branch[-1].bias, 0)

    def forward(self, x_lr, scale, infer_stoc:bool=False):
        B, _, L_lr = x_lr.shape
        L_hr = int(L_lr * scale)
        gen_noise = infer_stoc or self.training
        
        macro_feat = self.macro_proj(self.macro_encoder(x_lr))
        macro_feat = F.interpolate(macro_feat, size=L_lr, mode='nearest')
        micro_feat = self.micro_encoder(x_lr)
        
        z = torch.cat([macro_feat, micro_feat], dim=1)
        z_pad = F.pad(z, (1, 1), mode='replicate')
        z_prev = z_pad[:, :, :-2]
        z_curr = z_pad[:, :, 1:-1]
        z_next = z_pad[:, :, 2:]
        z_triplet = torch.cat([z_prev, z_curr, z_next], dim=1).transpose(1, 2)
        
        scale_tensor = torch.ones(B, L_lr, 1, device=x_lr.device, dtype=x_lr.dtype) * scale
        feat = torch.cat([scale_tensor, z_triplet], dim=-1)
        
        alphas = self.alpha_branch(feat)
        betas = self.beta_branch(feat)
        gammas = self.gamma_branch(feat)
        
        if gen_noise and not self.training: gammas *= temperature
        
        t_hr = torch.arange(L_hr, device=x_lr.device).float() / scale
        t_hr = t_hr.unsqueeze(0).repeat(B, 1)
        
        if self.training and do_perturbation:
            eta = torch.rand_like(t_hr) - 0.5
            t_select = t_hr + eta
        else:
            t_select = t_hr
            
        idx_i = torch.round(t_select).long().clamp(0, L_lr - 1)
        t_rel = (t_hr - idx_i.float()).unsqueeze(-1)
        
        freq_exps = torch.arange(num_bands, device=x_lr.device, dtype=torch.float32)
        frequencies = self.omega * (2.0 ** freq_exps)
        frequencies = frequencies.view(1, 1, -1)
        t_scaled = t_rel * frequencies
        pe = torch.cat([torch.sin(t_scaled), torch.cos(t_scaled), t_rel], dim=-1)
        
        scale_tensor_hr = torch.ones(B, L_hr, 1, device=x_lr.device, dtype=x_lr.dtype) * scale
        canvas = torch.cat([pe, scale_tensor_hr], dim=-1)
        canvas = self.input_projection(canvas)
        
        idx_alpha_beta = idx_i.view(B, L_hr, 1, 1).expand(-1, -1, num_blocks, mdim)
        idx_gamma = idx_i.view(B, L_hr, 1, 1).expand(-1, -1, num_blocks, 1)
        
        gathered_alphas = torch.gather(alphas.view(B, L_lr, num_blocks, mdim), 1, idx_alpha_beta)
        gathered_betas = torch.gather(betas.view(B, L_lr, num_blocks, mdim), 1, idx_alpha_beta)
        gathered_gammas = torch.gather(gammas.view(B, L_lr, num_blocks, 1), 1, idx_gamma)
        
        for i in range(min(noisy_start,num_blocks) if gen_noise else num_blocks):
            alpha = gathered_alphas[:, :, i, :]
            beta = gathered_betas[:, :, i, :]
            canvas = self.affine_transformations[i](alpha, beta, 0, canvas, False)
        
        if noisy_start < num_blocks and gen_noise:
            
            dropA,dropB = 1,1
            pA,pB = canvas,canvas.clone()
            for i in range(noisy_start,num_blocks):
                
                alpha = gathered_alphas[:, :, i, :]
                beta = gathered_betas[:, :, i, :]
                gamma = gathered_gammas[:, :, i, :]
                
                if self.training and dropA == 1 and dropB == 1:
                    if torch.rand(1).item() < drop_prob: 
                        dropA = 0
                    elif torch.rand(1).item() < drop_prob: 
                        dropB = 0
                
                pA = self.affine_transformations[i](alpha, beta, gamma*dropA, pA)
                pB = self.affine_transformations[i](alpha, beta, gamma*dropB, pB)
                
            return self.output_block(pA).mT.contiguous() * self.opcs,self.output_block(pB).mT.contiguous() * self.opcs
        
        res = self.output_block(canvas).mT.contiguous() * self.opcs
        return res, res.clone()
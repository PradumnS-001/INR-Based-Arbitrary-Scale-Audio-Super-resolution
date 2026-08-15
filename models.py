import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import *
from extraUtils.layers import getActivation, WeightNormLinear
from encodec.modules import SEANetEncoder

class AffineTransformation(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.RMSNorm(mdim)
        self.fc1 = WeightNormLinear(mdim, mdim//2)
        self.fc2 = WeightNormLinear(mdim//2, mdim)
        self.actv = getActivation(actfd)
        
    def forward(self, alpha:torch.Tensor, beta:torch.Tensor, canvas:torch.Tensor):
        skip = canvas
        canvas = self.fc1(self.norm(canvas)) * alpha
        canvas = self.fc2(self.actv(canvas + beta))
        return skip + canvas
    
class ImprovedLISA(nn.Module):
    def __init__(self, opcs:torch.Tensor=torch.tensor(0.03)):
        super().__init__()
        
        self.macro_encoder = SEANetEncoder(n_filters=filters, dimension=mdim1, ratios=[4,4,4], lstm=0)
        self.register_buffer('opcs', opcs.clamp_(min=0.001))
        
        # Micro acts as the Canvas
        self.micro_encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            getActivation(act=actfe),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            getActivation(act=actfe),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            getActivation(act=actfe),
            nn.Conv1d(64, 32, kernel_size=1)
        )
        
        # Alpha/Beta Branches (Taking Macro Feat as is)
        self.alpha_branch = nn.Sequential(
            WeightNormLinear(mdim1, mdim1),
            getActivation(act=actfs),
            WeightNormLinear(mdim1, mdim1),
            getActivation(act=actfs),
            WeightNormLinear(mdim1, mdim1),
            getActivation(act=actfs),
            nn.Linear(mdim1, num_blocks * mdim // 2)
        )
        
        self.beta_branch = nn.Sequential(
            WeightNormLinear(mdim1, mdim1),
            getActivation(act=actfs),
            WeightNormLinear(mdim1, mdim1),
            getActivation(act=actfs),
            WeightNormLinear(mdim1, mdim1),
            getActivation(act=actfs),
            nn.Linear(mdim1, num_blocks * mdim // 2)
        )
        
        k = (32 * 3 + 2 + mdim) // 2
        self.input_projection = nn.Sequential(
            nn.Linear(32 * 3 + 2, k // 2), 
            getActivation(actfd),
            nn.Linear(k // 2, mdim)
        )
        
        self.affine_transformations = nn.ModuleList([
            AffineTransformation() for _ in range(num_blocks)
        ])
        
        self.output_block = nn.Sequential(
            nn.Linear(mdim, mdim // 2),
            getActivation(actfd),
            nn.Linear(mdim // 2, 1)
        )
        
        nn.init.uniform_(self.alpha_branch[-1].bias, a=0.9, b=1.1)
        nn.init.uniform_(self.beta_branch[-1].bias, a=-0.1, b=0.1)

    def forward(self, x_lr, scale):
        B, _, L_lr = x_lr.shape
        L_hr = int(L_lr * scale)
        
        # 1. Macro generates context for Alpha/Beta as is
        macro_feat = self.macro_encoder(x_lr).mT
        alphas = self.alpha_branch(macro_feat).mT
        betas = self.beta_branch(macro_feat).mT
        
        alphas = F.interpolate(alphas, size=L_lr, mode='nearest').mT
        betas = F.interpolate(betas, size=L_lr, mode='nearest').mT
        # 2. Micro generates the continuous Canvas (with 3-neighbor context)
        micro_feat = self.micro_encoder(x_lr)
        
        m_pad = F.pad(micro_feat, (1, 1), mode='replicate')
        m_prev = m_pad[:, :, :-2]
        m_curr = m_pad[:, :, 1:-1]
        m_next = m_pad[:, :, 2:]
        
        # Concat the 3 latents and transpose for gathering
        micro_feat_3 = torch.cat([m_prev, m_curr, m_next], dim=1).transpose(1, 2)
        
        # Scale to HR indices and apply perturbation
        t_hr = torch.arange(L_hr, device=x_lr.device).float() / scale
        if self.training and do_perturbation:
            eta = torch.randn_like(t_hr) * 0.4
            t_select = t_hr + eta
        else:
            t_select = t_hr
            
        idx_i = torch.floor(t_select).long().clamp(0, L_lr - 1)
        t_rel = (t_hr - idx_i.float()).view(1, L_hr, 1).expand(B, -1, -1)
        
        # Gather Alpha, Beta, and Canvas to High-Res
        idx_alpha_beta = idx_i.view(1, L_hr, 1, 1).expand(B, -1, num_blocks, mdim//2)
        idx_canvas = idx_i.view(1, L_hr, 1).expand(B, -1, 32 * 3)
        
        gathered_alphas = torch.gather(alphas.view(B, L_lr, num_blocks, mdim//2), 1, idx_alpha_beta)
        gathered_betas = torch.gather(betas.view(B, L_lr, num_blocks, mdim//2), 1, idx_alpha_beta)
        gathered_canvas = torch.gather(micro_feat_3, 1, idx_canvas)
        
        scale_tensor_hr = torch.ones(B, L_hr, 1, device=x_lr.device, dtype=x_lr.dtype) * scale
        
        # Initialize Canvas with all perturbed components
        canvas = torch.cat([gathered_canvas, scale_tensor_hr, t_rel], dim=-1)
        canvas = self.input_projection(canvas)
        
        # 3. Affine Sculpting
        for i in range(num_blocks):
            alpha = gathered_alphas[:, :, i, :]
            beta = gathered_betas[:, :, i, :]
            canvas = self.affine_transformations[i](alpha, beta, canvas)
        
        res = self.output_block(canvas).mT.contiguous() * self.opcs
        return res
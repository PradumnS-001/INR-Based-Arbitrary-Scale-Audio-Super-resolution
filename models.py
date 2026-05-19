import torch
import torch.nn as nn
import torch.nn.functional as F

class LISA(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=1)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(1 + 3 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

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
        
        feat = torch.cat([t_rel, z_triplet], dim=-1)
        out = self.decoder(feat)
        
        return out.transpose(1, 2)
import torch
from torch import nn
import torch.nn.functional as F 

class MultiScaleSpectralLoss(nn.Module):
    """
    Implements the Multi-resolution STFT loss.
    Consists of Spectral Convergence (L2) and Log STFT Magnitude (L1) losses.
    """
    def __init__(self, n_ffts=[512, 1024, 2048]):
        super().__init__()
        self.n_ffts = n_ffts

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x_hat = x_hat.squeeze(1)
        x = x.squeeze(1)
        
        total_loss = 0
        for n in self.n_ffts:
            hop = n // 4
            window = torch.hann_window(n, device=x.device)
            
            s_hat = torch.stft(x_hat, n, hop_length=hop, window=window, return_complex=True).abs()
            s = torch.stft(x, n, hop_length=hop, window=window, return_complex=True).abs()
            
            sc_loss = torch.norm(s - s_hat, p="fro") / torch.norm(s, p="fro").clamp(min=1e-7)
            
            mag_loss = F.l1_loss(torch.log(s_hat + 1e-7), torch.log(s + 1e-7))
            
            total_loss += (sc_loss + mag_loss)
            
        return total_loss
    
def log_spectral_distance(y_hat:torch.Tensor, y:torch.Tensor)->torch.Tensor:
    
    """
    Measures the log spectral distance
    """
    
    n_fft = 512
    s_hat = torch.stft(y_hat.squeeze(1), n_fft, return_complex=True, window=torch.hann_window(n_fft, device=y.device)).abs().pow(2)
    s = torch.stft(y.squeeze(1), n_fft, return_complex=True,window=torch.hann_window(n_fft, device=y.device)).abs().pow(2)
    
    log_s_hat = 10*torch.log10(s_hat + 1e-7)
    log_s = 10*torch.log10(s + 1e-7)
    
    dist = torch.sqrt(torch.mean((log_s - log_s_hat)**2, dim=-2))
    return torch.mean(dist)
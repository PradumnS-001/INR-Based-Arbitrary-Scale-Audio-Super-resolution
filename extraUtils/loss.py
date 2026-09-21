import torch
from torch import nn
import torch.nn.functional as F 

def ganin_scheduler(epoch):
    if epoch == 0: return 0
    return torch.tanh(torch.tensor(epoch/2 - 0.5)).item()

def kullback_liebler_divergence(mean:torch.Tensor, std:torch.Tensor):
    return (-0.5 * (1 + torch.log(std**2) - mean**2 - std**2)).mean()

class MultiScaleSpectralLoss(nn.Module):
    """
    Implements the Multi-resolution STFT loss.
    Consists of Spectral Convergence (L2) and Power (Huber) losses.
    """
    def __init__(self, n_ffts=[2048, 512, 128]):
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
            
            mag_loss = F.huber_loss(torch.sqrt(s_hat + 1e-5), torch.sqrt(s + 1e-5), delta=0.25)
            
            total_loss += (sc_loss + mag_loss)
            
        return total_loss
    
def log_spectral_distance(y_hat:torch.Tensor, y:torch.Tensor, n_fft = 512)->torch.Tensor:
    """
    Measures the log spectral distance
    """
    window = torch.hann_window(n_fft, device=y.device)
        
    s_hat = torch.stft(y_hat.squeeze(1) if y_hat.ndim > 1 else y_hat, 
                        n_fft, return_complex=True, window=window).abs().pow(2)
    s = torch.stft(y.squeeze(1) if y.ndim > 1 else y, 
                    n_fft, return_complex=True, window=window).abs().pow(2)
    
    log10_s_hat = torch.log10(s_hat + 1e-10)
    log10_s = torch.log10(s + 1e-10)
    
    dist_per_frame = torch.sqrt(torch.mean((log10_s - log10_s_hat)**2, dim=-2))
    return torch.mean(dist_per_frame)
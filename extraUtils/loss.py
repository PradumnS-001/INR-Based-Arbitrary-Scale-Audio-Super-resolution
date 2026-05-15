import torch
from torch import nn
import torch.nn.functional as F 

class MultiScaleSpectralLoss(nn.Module):
    
    """
    Calculates ( log |Y_hat| - log |Y| )**2
    """
    
    def __init__(self, n_ffts=[512, 1024, 2048]):
        super().__init__()
        self.n_ffts = n_ffts

    def forward(self, x_hat:torch.Tensor, x:torch.Tensor)->torch.Tensor:
        loss = 0
        for n in self.n_ffts:
            s_hat = torch.stft(x_hat.squeeze(1), n, hop_length=n//4, 
                               window=torch.hann_window(n, device=x.device), 
                               return_complex=True).abs()
            s = torch.stft(x.squeeze(1), n, hop_length=n//4, 
                           window=torch.hann_window(n, device=x.device), 
                           return_complex=True).abs()
            loss += F.l1_loss(torch.log(s_hat + 1e-7), torch.log(s + 1e-7))
            
        return loss
    
def log_spectral_distance(y_hat:torch.Tensor, y:torch.Tensor)->torch.Tensor:
    
    """
    Measures the log spectral distance
    """
    
    n_fft = 2048
    s_hat = torch.stft(y_hat.squeeze(1), n_fft, return_complex=True).abs().pow(2)
    s = torch.stft(y.squeeze(1), n_fft, return_complex=True).abs().pow(2)
    
    log_s_hat = torch.log10(s_hat + 1e-7)
    log_s = torch.log10(s + 1e-7)
    
    dist = torch.sqrt(torch.mean(10*(log_s - log_s_hat)**2, dim=-2))
    return torch.mean(dist)
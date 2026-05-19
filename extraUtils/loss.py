import torch
from torch import nn
import torch.nn.functional as F 

class MultiScaleSpectralLoss(nn.Module):
    """
    Implements the Multi-resolution STFT loss.
    Consists of Spectral Convergence (L2) and Log STFT Magnitude (L1) losses.
    """
    def __init__(self, n_ffts=[2048, 512]):
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
            
            mag_loss = F.l1_loss(torch.log(s_hat + 1e-5), torch.log(s + 1e-5))
            
            total_loss += (sc_loss + mag_loss)
            
        return total_loss
    
def log_spectral_distance(y_hat:torch.Tensor, y:torch.Tensor)->torch.Tensor:
    """
    Measures the log spectral distance
    """
    n_fft = 512
    s_hat = torch.stft(y_hat.squeeze(1), n_fft, return_complex=True, window=torch.hann_window(n_fft, device=y.device)).abs().pow(2)
    s = torch.stft(y.squeeze(1), n_fft, return_complex=True,window=torch.hann_window(n_fft, device=y.device)).abs().pow(2)
    
    log_s_hat = torch.log(s_hat + 1e-5)
    log_s = torch.log(s + 1e-5)
    
    dist = torch.sqrt(torch.mean((log_s - log_s_hat)**2, dim=-2))
    return torch.mean(dist)

def balance_grad_norm(
    model:nn.Module,
    losses:list[torch.Tensor],
    weights:list[float | None] | None=None,
    norms:float | list[float | None] | None=None,
    scalar:float=1
    )->float:
    """
    Balances, optionally clips, and accumulates gradients for multiple losses.
    If `norms` is a list/None, computes and clips gradients in a vacuum before weighting and summing.
    If `norms` is a float, weights and sums losses first, then computes and globally clips the total gradient.
    Note: `losses`, `weights` (if provided), and `norms` (if list) must be of the exact same length.
    """
    
    if weights is None:
        weights = [1] * len(losses)
    weights = [abs(i * scalar) if i is not None else 0 for i in weights]
    
    if not isinstance(norms, float):
        
        params = list(model.parameters())
        accumulated_grads = [None] * len(params)
        
        for i, loss in enumerate(losses):
            
            model.zero_grad(set_to_none=True)
            loss.backward(retain_graph=(i < len(losses) - 1))
            
            if norms is not None and i < len(norms) and norms[i] is not None:
                torch.nn.utils.clip_grad_norm_(parameters=params, max_norm=norms[i])
            
            with torch.no_grad():
                for idx, param in enumerate(params):
                    if param.grad is not None:
                        if accumulated_grads[idx] is None: accumulated_grads[idx] = param.grad.clone().mul_(weights[i])
                        else: accumulated_grads[idx].add_(param.grad, alpha=weights[i])
                    
        for p, g in zip(params, accumulated_grads):
            if g is not None:
                p.grad = g
                
    else:
        
        total_loss = sum([i * w for i, w in zip(losses, weights)])
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=norms)
        
    return sum([float(i.item() * w) for i, w in zip(losses, weights)])
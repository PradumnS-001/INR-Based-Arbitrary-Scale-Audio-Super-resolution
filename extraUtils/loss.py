import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_func

def ganin_scheduler(epoch):
    return torch.tanh(torch.tensor(epoch/2)).item()

class WaveLoss(nn.Module):
    def __init__(self, eps, pow_fac, n_ffts=[2048, 512, 128]):
        super().__init__()
        self.n_ffts = n_ffts
        self.eps = eps
        self.windows = nn.ParameterDict({
            str(n): nn.Parameter(torch.hann_window(n), requires_grad=False)
            for n in n_ffts
        })
        self.pow_fac = pow_fac

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor, scale: float) -> torch.Tensor:
        if x.ndim == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
            x_hat = x_hat.squeeze(1)
        mssl_loss = 0.0
        
        for n in self.n_ffts:
            hop = n // 4
            window = self.windows[str(n)]
            
            s_hat_abs = torch.stft(x_hat.float(), n, hop_length=hop, window=window.float(), return_complex=True).abs()
            s_abs = torch.stft(x.float(), n, hop_length=hop, window=window.float(), return_complex=True).abs()
            
            total_bins = s_abs.shape[1]
            cutoff_bin = int((total_bins - 1) / scale)
            s_hat_high = s_hat_abs[:, cutoff_bin:, :]
            s_abs_high = s_abs[:, cutoff_bin:, :]
            
            mag_diff = s_abs_high - s_hat_high
            sc_loss = torch.norm(mag_diff, p="fro") / torch.norm(s_abs_high, p="fro").clamp(min=self.eps)
            
            pow_s_hat = s_hat_high.clamp(self.eps).pow(self.pow_fac)
            pow_s = s_abs_high.clamp(self.eps).pow(self.pow_fac)
            mag_loss = F.mse_loss(pow_s_hat, pow_s)
            
            mssl_loss += (sc_loss + mag_loss)
        
        return mssl_loss
        
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
            
def compute_audio_ssim(waveform_pred: torch.Tensor, waveform_target: torch.Tensor, sample_rate=None):
    """
    Computes SSIM directly on the log-power STFT to perfectly match the LSD metric space.
    """
    n_fft = 512
    hop = n_fft // 4  # Standard overlap
    window = torch.hann_window(n_fft, device=waveform_pred.device)
    
    # 1. Compute exact same Power STFT as your LSD function
    s_p = torch.stft(waveform_pred.squeeze(1), n_fft, hop_length=hop, return_complex=True, window=window).abs().pow(2)
    s_t = torch.stft(waveform_target.squeeze(1), n_fft, hop_length=hop, return_complex=True, window=window).abs().pow(2)
    
    # 2. Log compression (matches LSD +1e-5 epsilon)
    log_s_p = torch.log(s_p + 1e-5)
    log_s_t = torch.log(s_t + 1e-5)
    
    # 3. Strict Normalization to [0, 1] bounded by the target's min/max
    min_v, max_v = log_s_t.min(), log_s_t.max()
    spec_p_norm = torch.clamp((log_s_p - min_v) / (max_v - min_v + 1e-5), 0.0, 1.0)
    spec_t_norm = torch.clamp((log_s_t - min_v) / (max_v - min_v + 1e-5), 0.0, 1.0)
    
    # 4. Shape for torchmetrics SSIM: requires [Batch, Channel, Height, Width]
    if spec_p_norm.ndim == 2: # [Freq, Time]
        spec_p_norm = spec_p_norm.unsqueeze(0).unsqueeze(0)
        spec_t_norm = spec_t_norm.unsqueeze(0).unsqueeze(0)
    elif spec_p_norm.ndim == 3: # [Batch, Freq, Time]
        spec_p_norm = spec_p_norm.unsqueeze(1)
        spec_t_norm = spec_t_norm.unsqueeze(1)
        
    return ssim_func(spec_p_norm, spec_t_norm, data_range=1.0).item()

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
import os
import torch
from torch import nn
import torch.nn.functional as F
import math

def get_leaf_files(
    path:str, 
    ender:str|tuple[str]='', 
    starter:str|tuple[str]='', 
    container:str='')->tuple[int,list[str]]:
    """
    Returns all the leaf-files in a folder
    """
    path = rf'{path}'
    files = []
    with os.scandir(path=path) as entries:
        
        for entry in entries:
            
            name = os.path.join(path, entry.name)
            if entry.is_file():
                
                files += [name] if (name.startswith(starter) and name.endswith(ender) and container in name) else []
                
            elif entry.is_dir():
                
                files += get_leaf_files(
                    path= name,
                    starter=starter,
                    ender=ender,
                    container=container)[1]
                
            else: files += []
            
    return len(files), files

def count_params(model:nn.Module):
    """Prints a brief summary of the trainable parameter count per module."""
    print("\n" + "="*45)
    print("Model Parameter Summary")
    print("="*45)
    
    total_params = 0
    module_counts = {}
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Group by the top-level module name (e.g., 'macro_encoder', 'alpha_branch')
            module_name = name.split('.')[0]
            module_counts[module_name] = module_counts.get(module_name, 0) + param.numel()
            total_params += param.numel()
            
    for mod, count in module_counts.items():
        print(f"{mod:<25}: {count:,}")
        
    print("-" * 45)
    print(f"{'Total Trainable Params':<25}: {total_params:,}")
    print("="*45 + "\n")
        
    return total_params

def _get_zero_phase_taper(L: int, sr: float, cutoff_freq: float, rolloff: float, device: torch.device) -> torch.Tensor:
    """
    Generates a zero-phase frequency-domain raised-cosine taper.
    """
    num_bins = L // 2 + 1
    freqs = torch.arange(num_bins, device=device) * (sr / L)
    taper = torch.ones_like(freqs)

    f_stop = cutoff_freq
    f_pass = cutoff_freq * rolloff

    taper[freqs >= f_stop] = 0.0

    if f_stop > f_pass:
        transition = (freqs > f_pass) & (freqs < f_stop)
        taper[transition] = 0.5 * (1.0 + torch.cos(torch.pi * (freqs[transition] - f_pass) / (f_stop - f_pass)))

    return taper

def filter_fft(x: torch.Tensor, sr: float, cutoff_freq: float, rolloff: float = 0.9) -> torch.Tensor:
    """
    Zero-phase, zero-delay lowpass filter using frequency-domain tapering.
    Useful for filtering without changing the sample rate.
    """
    if cutoff_freq >= sr / 2.0:
        return x  # Nothing to filter if cutoff is above Nyquist

    B, C, L = x.shape

    # ~25ms padding for reflection to prevent edge Gibbs ringing
    pad_samples = int(0.025 * sr)
    pad_samples = min(pad_samples, L - 1)  # Guard against short sequences
    pad_samples = max(1, pad_samples)

    # Reflection pad
    x_padded = F.pad(x, (pad_samples, pad_samples), mode='reflect')
    L_pad = x_padded.shape[-1]

    # FFT -> Apply taper -> iFFT
    X = torch.fft.rfft(x_padded, dim=-1)
    taper = _get_zero_phase_taper(L_pad, sr, cutoff_freq, rolloff, x.device)
    X_filtered = X * taper
    x_filtered_pad = torch.fft.irfft(X_filtered, n=L_pad, dim=-1)

    # Crop symmetric margins
    return x_filtered_pad[..., pad_samples : pad_samples + L]

def resample_fft(x: torch.Tensor, orig_freq: int, new_freq: int, rolloff: float = 0.9) -> torch.Tensor:
    """
    Zero-phase, zero-jitter resampling with exact sub-sample sequence length alignment
    and raised-cosine anti-aliasing taper.
    """
    if orig_freq == new_freq:
        return x

    B, C, L = x.shape
    new_L = int(round(L * new_freq / orig_freq))

    # 1. Align pad_samples to eliminate fractional sub-sample phase jitter
    gcd = math.gcd(orig_freq, new_freq)
    step_orig = orig_freq // gcd
    step_new = new_freq // gcd

    # Target ~25ms padding, rounded to an exact integer multiple of step_orig
    target_pad = int(0.025 * orig_freq)
    multiplier = max(1, target_pad // step_orig)
    pad_left = multiplier * step_orig

    # Guard against reflect pad overflow on short sequences
    if pad_left >= L:
        pad_left = max(1, (L - 1) // step_orig) * step_orig

    # FIX: Ensure total padded length is a perfect multiple of step_orig
    rem = (L + pad_left) % step_orig
    pad_right = pad_left if rem == 0 else pad_left + (step_orig - rem)

    # Calculate exact out-pad for precise cropping
    pad_out_left = pad_left * step_new // step_orig

    # Safe padding mode: PyTorch 'reflect' requires pad size < L. 
    # With min seq length 4k this is practically guaranteed, but fallback protects weird edge rates.
    pad_mode = 'reflect' if (pad_left < L and pad_right < L) else 'replicate'
    
    # 2. Apply asymmetric pad
    x_padded = F.pad(x, (pad_left, pad_right), mode=pad_mode)
    L_pad = x_padded.shape[-1]
    
    # new_L_pad is now guaranteed to be an EXACT integer, preserving pitch perfectly
    new_L_pad = L_pad * step_new // step_orig 

    X = torch.fft.rfft(x_padded, dim=-1)
    num_bins = X.shape[-1]

    if new_freq < orig_freq:
        # Downsampling: Taper high frequencies, then slice spectrum
        cutoff = new_freq / 2.0
        taper = _get_zero_phase_taper(L_pad, orig_freq, cutoff, rolloff, x.device)
        X = X * taper

        target_bins = new_L_pad // 2 + 1
        X_out = X[..., :target_bins].clone()

        # Ensure Nyquist bin is strictly real
        if new_L_pad % 2 == 0:
            X_out[..., -1] = torch.complex(X_out[..., -1].real, torch.zeros_like(X_out[..., -1].real))

    else:
        # Upsampling: Taper near the original Nyquist, then zero-pad spectrum
        cutoff = orig_freq / 2.0
        taper = _get_zero_phase_taper(L_pad, orig_freq, cutoff, rolloff, x.device)
        taper[-1] = 0.0 # Force Nyquist to 0 before extending
        X = X * taper

        target_bins = new_L_pad // 2 + 1
        X_out = torch.zeros(*X.shape[:-1], target_bins, dtype=X.dtype, device=x.device)
        # Fast in-place assignment (safe since this isn't tracking autograd)
        X_out[..., :num_bins] = X

    # 3. Inverse FFT with energy scale correction
    x_new_pad = torch.fft.irfft(X_out, n=new_L_pad, dim=-1) * (new_L_pad / L_pad)

    # 4. Crop using exact left margin alignment to retrieve final sequence
    return x_new_pad[..., pad_out_left : pad_out_left + new_L]
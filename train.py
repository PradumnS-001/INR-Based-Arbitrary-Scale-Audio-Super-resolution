import torch
from torch.nn import functional as F
import torchaudio
from tqdm import tqdm
import numpy as np
import os

from data import tr_loader, val_loader
from configs import *
from models import LISA
from extraUtils.loss import MultiScaleSpectralLoss, log_spectral_distance
from torchmetrics.audio import SignalNoiseRatio

def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    torch.backends.cudnn.benchmark = False

    mssl = MultiScaleSpectralLoss()
    snr_metric = SignalNoiseRatio().to(device)

    model = LISA().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    # ==============================================================================
    # Modern Standard: Cached Resample Filter Banks with Float-less Math
    # ==============================================================================
    print("Precomputing Resample Filter Banks...")
    resamplers = {}
    
    max_scale_int = (scale_res * max_target_sr) // low_sampling_rate
    
    for s in range(scale_res, max_scale_int + 1):
        # Multiply FIRST, then floor divide to bypass floating-point precision drift
        target_freq = (low_sampling_rate * s) // scale_res
        
        # Key the dictionary directly by the integer 's'
        resamplers[s] = torchaudio.transforms.Resample(
            orig_freq=high_sampling_rate, new_freq=target_freq
        ).to(device=device, non_blocking=True)
        
    resample_to_min = torchaudio.transforms.Resample(
        orig_freq=high_sampling_rate, new_freq=low_sampling_rate
    ).to(device=device, non_blocking=True)

    print("Training Started")

    for epoch in range(epochs):
        
        model.train()
        epoch_loss = 0
        pbar = tqdm(tr_loader, desc=f"Epoch {epoch}")
        
        # Modern Standard: Ignore the pre-degraded lr_wav to prevent filter compounding
        for step, (_, hr_wav) in enumerate(pbar):
            
            hr_wav = hr_wav.to(device, non_blocking=True)
            
            # Sample integer scale using the approximator
            s = np.random.randint(scale_res, max_scale_int + 1)
            target_sr = (low_sampling_rate * s) // scale_res
            true_scale = target_sr / low_sampling_rate
            
            # Parallel native resampling directly from the pristine HR source
            lr_input = resample_to_min(hr_wav)
            hr_target = resamplers[s](hr_wav)
            
            optimizer.zero_grad()
            
            pred = model(lr_input, scale=true_scale)
            min_len = min(pred.shape[-1], hr_target.shape[-1])
            pred, hr_target = pred[..., :min_len], hr_target[..., :min_len]
            
            loss = mssl_wt * mssl(pred, hr_target) + l1_wt * F.l1_loss(pred, hr_target)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()
            
            epoch_loss += loss.detach()
            if step % 10 == 0:
                pbar.set_postfix({"loss": loss.item()})
            
        torch.cuda.empty_cache()
        
        avg_train_loss = epoch_loss / len(tr_loader)
        
        model.eval()
        snr_metric.reset()
        avg_lsd = 0
        
        with torch.no_grad():
            for _, hr_wav in val_loader:
                hr_wav = hr_wav.to(device)
                
                s_val = int(val_scale * scale_res)
                
                # Fetch target directly from cache if possible
                lr_input = resample_to_min(hr_wav)
                if s_val in resamplers:
                    hr_target = resamplers[s_val](hr_wav)
                    target_sr = (low_sampling_rate * s_val) // scale_res
                    true_scale = target_sr / low_sampling_rate
                else:
                    target_sr = int(low_sampling_rate * val_scale)
                    hr_target = torchaudio.functional.resample(hr_wav, high_sampling_rate, target_sr).to(device)
                    true_scale = val_scale
                
                pred = model(lr_input, scale=true_scale)
                min_len = min(pred.shape[-1], hr_target.shape[-1])
                pred, hr_target = pred[..., :min_len], hr_target[..., :min_len]
                
                snr_metric(pred, hr_target)
                avg_lsd += log_spectral_distance(pred, hr_target).item()
                
        current_val_snr = snr_metric.compute().item()
        current_val_lsd = avg_lsd / len(val_loader)
        
        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val SNR: {current_val_snr:.2f} | Val LSD: {current_val_lsd:.4f}")
        scheduler.step()

        torch.save(model.state_dict(), os.path.join('models', f"lisa_final_epoch.pt"))
        
if __name__ == "__main__":
    main()
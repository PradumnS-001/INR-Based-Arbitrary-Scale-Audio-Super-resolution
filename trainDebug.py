import torch
from torch.nn import functional as F
from torchaudio.functional import resample
import numpy as np

from data import tr_loader
from configs import *
from models import ImprovedLISA
from extraUtils.loss import WaveLoss, ModelEMA

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    supported = torch.cuda.is_bf16_supported()
    print(supported)
    cast_type = torch.bfloat16 if supported else torch.float16
    torch.backends.cudnn.benchmark = False

    waveloss = WaveLoss().to(device)
    model = ImprovedLISA().to(device)
    ema = ModelEMA(model=model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wdc)
    
    frozen_batches = []
    
    with torch.no_grad():
        for i, (lr_wav, hr_wav) in enumerate(tr_loader):
            lr_wav = lr_wav.to(device, non_blocking=True)
            hr_wav = hr_wav.to(device, non_blocking=True)
            scale = np.random.randint(50, int(50*high_sampling_rate/low_sampling_rate + 1)) / 50
            hsr_new = int(low_sampling_rate * scale)
            hr_wav = resample(hr_wav, high_sampling_rate, hsr_new)
            lr_wave_base = F.interpolate(lr_wav, size=hsr_new, mode='linear', align_corners=True)
            
            frozen_batches.append((lr_wav, hr_wav, lr_wave_base, scale))
            if len(frozen_batches) == update_step:
                break
                
    print(f"Captured {len(frozen_batches)} batches for overfitting.")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad(set_to_none=True)
        
        for i, (lr_wav, hr_wav, lr_wave_base, scale) in enumerate(frozen_batches):
            with torch.autocast(device_type=device.split(':')[0], dtype=cast_type):
                pred, gamma_loss = model(lr_wav, scale=scale)
                min_len = min(pred.shape[-1], hr_wav.shape[-1])
                pred, hr_wav_c = pred[...,:min_len], hr_wav[...,:min_len]
                lr_wave_base_c = lr_wave_base[...,:min_len]
                pred += lr_wave_base_c
                
                wl = waveloss(pred, hr_wav_c)
                l1_penalty = F.l1_loss(pred, hr_wav_c)
                loss:torch.Tensor = mssl_wt * wl[0] + l1_wt * l1_penalty + var_wt * wl[1] + g_wt * gamma_loss
            
            loss.backward()
            epoch_loss += loss.item()
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()
        ema.update(model)
        
        print(f"Epoch {epoch} | Train Loss: {epoch_loss:.4f}")

if __name__ == "__main__":
    main()
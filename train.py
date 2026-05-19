import torch
from torch.nn import functional as F
from torchaudio.functional import resample
from tqdm import tqdm
import numpy as np
import gc
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
    
    best_lsd = float('inf')
    best_snr = -1*float('inf')

    for epoch in range(epochs):
        
        model.train()
        epoch_loss = 0
        pbar = tqdm(tr_loader, desc=f"Epoch {epoch}")
        
        for lr_wav, hr_wav in pbar:
            
            lr_wav:torch.Tensor = lr_wav.to(device)
            scale = np.random.randint(50, int(50*high_sampling_rate/low_sampling_rate + 1)) / 50
            hsr_new = int(low_sampling_rate * scale)
            with torch.no_grad(): hr_wav = resample(hr_wav, high_sampling_rate, hsr_new)
            
            hr_wav:torch.Tensor = hr_wav.to(device)
            optimizer.zero_grad()
            
            pred = model(lr_wav, scale=scale)
            min_len = min(pred.shape[-1],hr_wav.shape[-1])
            pred, hr_wav = pred[...,:min_len], hr_wav[...,:min_len]
            
            loss:torch.Tensor = mssl_wt * mssl(pred, hr_wav) + l1_wt * F.l1_loss(pred, hr_wav)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            
        gc.collect()
        torch.cuda.empty_cache()
        
        avg_train_loss = epoch_loss / len(tr_loader)
        
        model.eval()
        snr_metric.reset()
        avg_lsd = 0
        with torch.no_grad():
            for lr_wav, hr_wav in val_loader:
                lr_wav, hr_wav = lr_wav.to(device), hr_wav.to(device)
                
                scale = val_scale
                hsr_new = int(low_sampling_rate * scale)
                hr_wav = resample(hr_wav, high_sampling_rate, hsr_new)
                
                pred = model(lr_wav, scale=scale)
                min_len = min(pred.shape[-1],hr_wav.shape[-1])
                pred, hr_wav = pred[...,:min_len], hr_wav[...,:min_len]
                
                snr_metric(pred, hr_wav)
                avg_lsd += log_spectral_distance(pred, hr_wav).item()
                
        current_val_snr = snr_metric.compute().item()
        current_val_lsd = avg_lsd / len(val_loader)
        
        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val SNR: {current_val_snr:.2f} | Val LSD: {current_val_lsd:.4f}")
        
        if current_val_lsd <= best_lsd:
            best_lsd = current_val_lsd
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lsd': best_lsd,
                'snr': current_val_snr
            }, os.path.join('models',"lisa_best_model_lsd.pt"))
            print(f"--> Best model saved with LSD: {best_lsd:.4f}")
        if current_val_snr >= best_snr:
            best_snr = current_val_snr
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lsd': current_val_lsd,
                'snr': best_snr
            }, os.path.join('models',"lisa_best_model_snr.pt"))
            print(f"--> Best model saved with SNR: {best_snr:.4f}")
        scheduler.step()
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join('models',f"lisa_checkpoint_epoch_{epoch}.pt"))
        
if __name__ == "__main__":
    main()
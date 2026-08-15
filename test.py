import torch
from torchaudio.functional import resample
import os
import soundfile as sf
from tqdm import tqdm
from torchmetrics.audio import SignalNoiseRatio, SignalDistortionRatio

from data import val_loader
from configs import *
from models import ImprovedLISA
from extraUtils.loss import log_spectral_distance, compute_audio_ssim

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing on {device}")
    
    # 1. Initialize Model & Load Weights
    model = ImprovedLISA(opcs=torch.tensor(0.03)).to(device)
    checkpoint_path = os.path.join('models', 'lisa_best_model_lsd.pth')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Cannot find {checkpoint_path}. Train the model first.")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully.")

    # 2. Setup Metrics
    snr_metric = SignalNoiseRatio().to(device)
    sdr_metric = SignalDistortionRatio().to(device)
    
    avg_lsd = 0
    avg_ssim = 0
    avg_base_lsd = 0
    
    os.makedirs('audio/test_results', exist_ok=True)

    # 3. Evaluation Loop
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Testing")
        for i, (lr_wav, hr_wav) in enumerate(pbar):
            lr_wav, hr_wav = lr_wav.to(device), hr_wav.to(device)
            
            scale = val_scale
            hsr_new = int(low_sampling_rate * scale)
            hr_wav = resample(hr_wav, high_sampling_rate, hsr_new)
            lr_wave_base = resample(lr_wav, low_sampling_rate, hsr_new)
            
            # Forward Pass
            pred = model(lr_wav, scale=scale)
            
            min_len = min(pred.shape[-1], hr_wav.shape[-1])
            pred, hr_wav = pred[...,:min_len], hr_wav[...,:min_len]
            lr_wave_base = lr_wave_base[...,:min_len]
            
            # Add Residual
            pred += lr_wave_base
            
            # Compute Metric Accumulations
            avg_lsd += log_spectral_distance(pred, hr_wav).item()
            avg_ssim += compute_audio_ssim(waveform_pred=pred, waveform_target=hr_wav, sample_rate=hsr_new)
            avg_base_lsd += log_spectral_distance(lr_wave_base, hr_wav).item()
            
            snr_metric.update(pred.squeeze(1), hr_wav.squeeze(1))
            sdr_metric.update(pred.squeeze(1), hr_wav.squeeze(1))
            
            # Save the first 3 samples for auditory inspection
            if i < 3:
                sf.write(os.path.join('audio/test_results', f"test_target_{i}.wav"), hr_wav.cpu()[0,...].mT.numpy(), hsr_new)
                sf.write(os.path.join('audio/test_results', f"test_base_{i}.wav"), lr_wave_base.cpu()[0,...].mT.numpy(), hsr_new)
                
                save = pred.cpu()[0,...].mT
                save /= max(save.abs().max().item(), 1)
                sf.write(os.path.join('audio/test_results', f"test_pred_{i}.wav"), save.contiguous().numpy(), hsr_new)

    # 4. Final Aggregation
    num_samples = len(val_loader)
    final_lsd = avg_lsd / num_samples
    final_ssim = avg_ssim / num_samples
    final_base_lsd = avg_base_lsd / num_samples
    final_snr = snr_metric.compute().item()
    final_sdr = sdr_metric.compute().item()
    
    print("\n" + "="*40)
    print("FINAL TEST SET RESULTS")
    print("="*40)
    print("Val Scale: ", val_scale)
    print(f"LSD:  {final_lsd:.4f} (Base Resampler: {final_base_lsd:.4f})")
    print(f"SSIM: {final_ssim:.4f}")
    print(f"SNR:  {final_snr:.4f} dB")
    print(f"SDR:  {final_sdr:.4f} dB")
    print("="*40)

if __name__ == "__main__":
    main()
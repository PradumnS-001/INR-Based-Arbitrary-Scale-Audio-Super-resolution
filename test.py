import os
import torch
import torchaudio
import soundfile as sf
import matplotlib.pyplot as plt
from torchaudio.functional import resample
from tqdm import tqdm
import numpy as np

from data import val_loader, val12_loader
from configs import *
from models import Encoder, INRDecoder
from extraUtils.loss import log_spectral_distance
from torchmetrics.audio import SignalNoiseRatio
from evaluation import Evaluator

def plot_mel_spectrogram(y_lr, y_true, y_mean, y_sample, lr_sr, hr_sr, save_path):
    mel_hr = torchaudio.transforms.MelSpectrogram(sample_rate=hr_sr, n_mels=80, n_fft=1024)
    db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)

    # Upsample LR so it maps perfectly to the HR mel scale for visual comparison
    y_lr_up = resample(y_lr.cpu(), lr_sr, hr_sr)
    
    mel_l = db_transform(mel_hr(y_lr_up)).squeeze().numpy()
    mel_t = db_transform(mel_hr(y_true.cpu())).squeeze().numpy()
    mel_m = db_transform(mel_hr(y_mean.cpu())).squeeze().numpy()
    mel_s = db_transform(mel_hr(y_sample.cpu())).squeeze().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    
    im0 = axes[0].imshow(mel_l, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title(f'Low Res Input Upsampled')
    axes[0].set_ylabel('Mel bins')
    axes[0].set_xlabel('Frames')
    
    im1 = axes[1].imshow(mel_t, aspect='auto', origin='lower', cmap='viridis')
    axes[1].set_title('Ground Truth (Target HR)')
    axes[1].set_xlabel('Frames')
    
    im2 = axes[2].imshow(mel_m, aspect='auto', origin='lower', cmap='viridis')
    axes[2].set_title('Deterministic Mean (z = mu)')
    axes[2].set_xlabel('Frames')
    
    im3 = axes[3].imshow(mel_s, aspect='auto', origin='lower', cmap='viridis')
    axes[3].set_title('Stochastic Sample (z = mu + eps*std)')
    axes[3].set_xlabel('Frames')

    fig.colorbar(im3, ax=axes.ravel().tolist(), format="%+2.0f dB")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def evaluate_and_save(model_path, model_name, device, num_runs=10):
    print(f"\n" + "="*55)
    print(f"--- Evaluating {model_name} ---")
    print("="*55)
    
    checkpoint = torch.load(model_path, map_location=device)
    encoder = Encoder().to(device)
    encoder.load_state_dict(checkpoint['encoder'])
    
    dummy_decoder = INRDecoder().to(device) 
    if 'ema' in checkpoint:
        dummy_decoder.load_state_dict(checkpoint['ema'])
    else:
        dummy_decoder.load_state_dict(checkpoint['decoder'])
        
    decoder = dummy_decoder
    
    # Trust the EMA checkpoint buffers completely to prevent amplitude blowout
    encoder.eval()
    decoder.eval()

    snr_metric = SignalNoiseRatio().to(device)
    evaluator = Evaluator()
    
    print(f"\n[Phase 1] Evaluating Regression to Mean & Standard Val Set...")
    snr_metric.reset()
    total_mean_lsd = 0.0
    total_expected_lsd = 0.0
    total_stochastic_variance = 0.0

    with torch.no_grad():
        for lr_wav, hr_wav in tqdm(val_loader, desc=f"Standard Val"):
            lr_wav, hr_wav = lr_wav.to(device), hr_wav.to(device)
            hsr_new = int(low_sampling_rate * val_scale)
            hr_wav = resample(hr_wav, high_sampling_rate, hsr_new).float()
            
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                mu, std = encoder(lr_wav)
                
                # 1. Deterministic Prediction (Mean)
                pred_mean = decoder(mu, scale=val_scale).float()
                min_len = min(pred_mean.shape[-1], hr_wav.shape[-1])
                pred_mean = pred_mean[..., :min_len]
                total_mean_lsd += log_spectral_distance(pred_mean, hr_wav[..., :min_len]).item()
            
            batch_avg_lsd = 0.0
            stochastic_preds = []
            
            # 2. Stochastic Predictions (Variance Testing)
            for _ in range(num_runs):
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    B, D, _ = std.shape if len(std.shape) == 3 else (1, std.shape[0], std.shape[1])
                    eps = torch.randn(B, D, 1).to(device)
                    z = mu + eps * std
                    pred_sample = decoder(z, scale=val_scale).float()
                
                pred_sample = pred_sample[..., :min_len]
                stochastic_preds.append(pred_sample)
                batch_avg_lsd += log_spectral_distance(pred_sample, hr_wav[..., :min_len]).item()
                
                if _ == 0: 
                    snr_metric(pred_sample, hr_wav[..., :min_len])

            # Measure variance across the 10 stochastic runs per sample
            stacked_preds = torch.stack(stochastic_preds, dim=0) # [10, B, 1, L]
            pixel_variance = torch.var(stacked_preds, dim=0).mean().item()
            total_stochastic_variance += pixel_variance
            
            total_expected_lsd += (batch_avg_lsd / num_runs)

    num_batches = len(val_loader)
    print(f"\nPhase 1 Results for {model_name}:")
    print(f"Standard Val SNR: {snr_metric.compute().item():.2f} dB")
    print(f"Deterministic (Mean) LSD: {total_mean_lsd / num_batches:.4f}")
    print(f"Stochastic Expected LSD: {total_expected_lsd / num_batches:.4f}")
    print(f"Regression Variance Test: {total_stochastic_variance / num_batches:.8f}")
    
    if (total_stochastic_variance / num_batches) < 1e-6:
        print(">>> WARNING: Variance is mathematically zero. The model suffered posterior collapse and regressed to the mean.")

    print(f"\n[Phase 2] Generating 12 Full-Length Samples & Saving Artifacts...")
    snr_metric.reset()
    
    total_12_expected_lsd = 0.0
    visqol_scores = []
    pesq_scores = []

    with torch.no_grad():
        for idx, (lr_wav, hr_wav) in enumerate(tqdm(val12_loader, desc=f"Full-Length 12")):
            lr_wav, hr_wav = lr_wav.to(device), hr_wav.to(device)
            hsr_new = int(low_sampling_rate * val_scale)
            hr_wav = resample(hr_wav, high_sampling_rate, hsr_new).float()
            
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                mu, std = encoder(lr_wav)
                
                # Mean Prediction
                pred_mean = decoder(mu, scale=val_scale).float()
                min_len = min(pred_mean.shape[-1], hr_wav.shape[-1])
                pred_mean = pred_mean[..., :min_len]
            
            # Stochastic Run
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                B, D, _ = std.shape if len(std.shape) == 3 else (1, std.shape[0], std.shape[1])
                eps = torch.randn(B, D, 1).to(device)
                z = mu + eps * std
                pred_sample = decoder(z, scale=val_scale).float()[..., :min_len]
                
            hr_eval = hr_wav[..., :min_len]
            
            total_12_expected_lsd += log_spectral_distance(pred_sample, hr_eval).item()
            snr_metric(pred_sample, hr_eval)
            
            # Evaluation metrics
            hr_list = [hr_eval.squeeze(0)]
            sr_list = [pred_sample.squeeze(0)]
            
            v_score = evaluator.evaluate_visqol(hr_list, sr_list, current_sr=hsr_new)
            p_score = evaluator.evaluate_pesq(hr_list, sr_list, current_sr=hsr_new)
            
            if not np.isnan(v_score): visqol_scores.append(v_score)
            if not np.isnan(p_score): pesq_scores.append(p_score)

            # File Saves
            lr_save_path = os.path.join('audio', f"{model_name}_clip{idx:02d}_1_LR.wav")
            hr_save_path = os.path.join('audio', f"{model_name}_clip{idx:02d}_2_HR.wav")
            mean_save_path = os.path.join('audio', f"{model_name}_clip{idx:02d}_3_Mean.wav")
            sample_save_path = os.path.join('audio', f"{model_name}_clip{idx:02d}_4_Sample.wav")
            
            sf.write(lr_save_path, lr_wav[0].cpu().numpy().T, low_sampling_rate)
            sf.write(hr_save_path, hr_eval[0].cpu().numpy().T, hsr_new)
            sf.write(mean_save_path, pred_mean[0].cpu().numpy().T, hsr_new)
            sf.write(sample_save_path, pred_sample[0].cpu().numpy().T, hsr_new)
            
            img_save_path = os.path.join('image', f"{model_name}_mel_grid_{idx:02d}.png")
            plot_mel_spectrogram(lr_wav[0], hr_eval[0], pred_mean[0], pred_sample[0], 
                                 low_sampling_rate, hsr_new, img_save_path)

    print(f"\nPhase 2 Results for {model_name} (12 Full-Length Samples):")
    print(f"Subset SNR:  {snr_metric.compute().item():.2f} dB")
    print(f"Subset LSD:  {total_12_expected_lsd / len(val12_loader):.4f}")
    print(f"Subset PESQ (Wide-Band): {np.mean(pesq_scores):.4f}")
    print(f"Subset ViSQOL: {np.mean(visqol_scores):.4f}")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    os.makedirs('audio', exist_ok=True)
    os.makedirs('image', exist_ok=True)
    
    best_lsd_path = os.path.join('models', 'lisa_best_model_lsd.pt')
    last_model_path = os.path.join('models', 'lisa_last_model.pt')

    if os.path.exists(best_lsd_path):
        evaluate_and_save(best_lsd_path, 'Best_LSD_Model', device)
    else:
        print(f"Warning: Could not find {best_lsd_path}. Skipping.")

    if os.path.exists(last_model_path):
        evaluate_and_save(last_model_path, 'Last_Epoch_Model', device)
    else:
        print(f"Warning: Could not find {last_model_path}. Skipping.")

if __name__ == '__main__':
    main()
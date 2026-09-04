import os
import torch
import torchaudio
import soundfile as sf
import matplotlib.pyplot as plt
from torchaudio.functional import resample
from tqdm import tqdm
import numpy as np

from data import val_loader, val12_loader, tr_loader, calc_opcs
from configs import *
from models import Encoder, INRDecoder
from extraUtils.loss import log_spectral_distance
from torchmetrics.audio import SignalNoiseRatio

try:
    from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
except ImportError:
    raise ImportError("Please install PESQ support: pip install pesq torchmetrics[audio]")

def plot_mel_spectrogram(y_true, y_pred, sr, save_path):
    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_mels=80, n_fft=1024)
    db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)

    mel_true = db_transform(mel_transform(y_true.cpu())).squeeze().numpy()
    mel_pred = db_transform(mel_transform(y_pred.cpu())).squeeze().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im1 = axes[0].imshow(mel_true, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title('Ground Truth Mel Spectrogram')
    axes[0].set_ylabel('Mel bins')
    axes[0].set_xlabel('Frames')
    fig.colorbar(im1, ax=axes[0], format="%+2.0f dB")

    im2 = axes[1].imshow(mel_pred, aspect='auto', origin='lower', cmap='viridis')
    axes[1].set_title('Predicted (LISA) Mel Spectrogram')
    axes[1].set_xlabel('Frames')
    fig.colorbar(im2, ax=axes[1], format="%+2.0f dB")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def evaluate_and_save(model_path, model_name, device, num_runs=10):
    print(f"\n" + "="*45)
    print(f"--- Evaluating {model_name} ({num_runs}-Run Expected Value) ---")
    print("="*45)
    
    # Load state dict strictly to map the mean/std buffers correctly
    checkpoint = torch.load(model_path, map_location=device)
    
    encoder = Encoder().to(device)
    encoder.load_state_dict(checkpoint['encoder'])
    
    # Temporarily init decoder to get structure, then load buffers
    dummy_decoder = INRDecoder().to(device) 
    if 'ema' in checkpoint:
        dummy_decoder.load_state_dict(checkpoint['ema'])
    else:
        dummy_decoder.load_state_dict(checkpoint['decoder'])
        
    decoder = dummy_decoder
    mean_data, std_data = calc_opcs(tr_loader)
    mean_data = torch.tensor(mean_data)
    decoder.mean_data = mean_data
    decoder.std_data = std_data
    encoder.eval()
    decoder.eval()

    snr_metric = SignalNoiseRatio().to(device)
    
    print(f"\n[Phase 1] Evaluating Standard Val Set...")
    snr_metric.reset()
    total_expected_lsd = 0.0

    with torch.no_grad():
        for lr_wav, hr_wav in tqdm(val_loader, desc=f"Standard Val"):
            lr_wav, hr_wav = lr_wav.to(device), hr_wav.to(device)
            hsr_new = int(low_sampling_rate * val_scale)
            hr_wav = resample(hr_wav, high_sampling_rate, hsr_new)
            hr_wav = hr_wav.float()
            
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                mu, std = encoder(lr_wav)
            
            batch_avg_lsd = 0.0
            
            # Execute 10 stochastic samples per batch to find the Expected Value
            for _ in range(num_runs):
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    if len(std.shape) == 3: B, D, _ = std.shape
                    else: 
                        D, _ = std.shape
                        B = 1
                    eps = torch.randn(B,D,1).to(device)
                    z = mu + eps * std
                    pred = decoder(z, scale=val_scale)
                    
                min_len = min(pred.shape[-1], hr_wav.shape[-1])
                pred = pred[..., :min_len].float()
                
                batch_avg_lsd += log_spectral_distance(pred, hr_wav[..., :min_len]).item()
                
                if _ == 0: # Only accumulate SNR for the first run to save computation
                    snr_metric(pred, hr_wav[..., :min_len])

            total_expected_lsd += (batch_avg_lsd / num_runs)

    final_snr_standard = snr_metric.compute().item()
    final_lsd_standard = total_expected_lsd / len(val_loader)
    
    print(f"\nPhase 1 Results for {model_name}:")
    print(f"Standard Val SNR: {final_snr_standard:.2f} dB")
    print(f"Standard Val Expected LSD: {final_lsd_standard:.4f}")

    print(f"\n[Phase 2] Evaluating 12 Full-Length Samples & Saving Artifacts...")
    snr_metric.reset()
    pesq_metric = PerceptualEvaluationSpeechQuality(fs=16000, mode='wb').to(device)
    pesq_metric.reset()
    
    total_12_expected_lsd = 0.0
    saved_audio_count = 0
    saved_img_count = 0

    with torch.no_grad():
        for idx, (lr_wav, hr_wav) in enumerate(tqdm(val12_loader, desc=f"Full-Length 12")):
            lr_wav, hr_wav = lr_wav.to(device), hr_wav.to(device)
            hsr_new = int(low_sampling_rate * val_scale)
            hr_wav = resample(hr_wav, high_sampling_rate, hsr_new).float()
            
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                mu, std = encoder(lr_wav)
            
            run_lsds = []
            run_preds = []
            
            # 10 stochastic runs for artifact saving
            for _ in range(num_runs):
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    if len(std.shape) == 3: B, D, _ = std.shape
                    else: 
                        D, _ = std.shape
                        B = 1
                    eps = torch.randn(B,D,1).to(device)
                    z = mu + eps * std
                    pred = decoder(z, scale=val_scale)
                    
                min_len = min(pred.shape[-1], hr_wav.shape[-1])
                pred = pred[..., :min_len].float()
                
                run_lsds.append(log_spectral_distance(pred, hr_wav[..., :min_len]).item())
                run_preds.append(pred)

            mean_run_lsd = np.mean(run_lsds)
            total_12_expected_lsd += mean_run_lsd
            
            # Select the generation closest to the mean LSD for honest visualization
            closest_idx = np.argmin(np.abs(np.array(run_lsds) - mean_run_lsd))
            rep_pred = run_preds[closest_idx]
            
            snr_metric(rep_pred, hr_wav[..., :min_len])
            
            hr_16k = resample(hr_wav[..., :min_len], hsr_new, 16000).squeeze(1)
            pred_16k = resample(rep_pred, hsr_new, 16000).squeeze(1)
            pesq_metric(pred_16k, hr_16k)

            if saved_audio_count < 4:
                lr_save_path = os.path.join('audio', f"{model_name}_sample{saved_audio_count}_LR.wav")
                pred_save_path = os.path.join('audio', f"{model_name}_sample{saved_audio_count}_SR.wav")
                hr_save_path = os.path.join('audio', f"{model_name}_sample{saved_audio_count}_HR.wav")
                
                lr_np = lr_wav[0].cpu().numpy()
                pred_np = rep_pred[0].cpu().numpy()
                hr_np = hr_wav[0, ..., :min_len].cpu().numpy()
                
                if lr_np.ndim == 2: lr_np = lr_np.T
                if pred_np.ndim == 2: pred_np = pred_np.T
                if hr_np.ndim == 2: hr_np = hr_np.T
                    
                sf.write(lr_save_path, lr_np, low_sampling_rate)
                sf.write(pred_save_path, pred_np, hsr_new)
                sf.write(hr_save_path, hr_np, hsr_new)
                
                saved_audio_count += 1
            
            elif saved_img_count < 2:
                img_save_path = os.path.join('image', f"{model_name}_mel_comparison_{saved_img_count}.png")
                plot_mel_spectrogram(hr_wav[0, ..., :min_len], rep_pred[0], hsr_new, img_save_path)
                saved_img_count += 1

    final_snr_12 = snr_metric.compute().item()
    final_lsd_12 = total_12_expected_lsd / len(val12_loader)
    final_pesq_12 = pesq_metric.compute().item()
    
    print(f"\nPhase 2 Results for {model_name} (12 Full-Length Samples):")
    print(f"Subset Representative SNR:  {final_snr_12:.2f} dB")
    print(f"Subset Expected LSD:  {final_lsd_12:.4f}")
    print(f"Subset Representative PESQ: {final_pesq_12:.4f} (Wide-Band)")

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
import os
import torch
import torchaudio
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from visqol import VisqolApi

# Data and model imports
from configs import *
from data import val100_loader
from models import Model

def plot_mel_spectrogram(y_true, y_pred, y_baseline, sr, save_path):
    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_mels=80, n_fft=1024)
    db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)

    mel_true = db_transform(mel_transform(y_true.cpu())).squeeze().numpy()
    mel_pred = db_transform(mel_transform(y_pred.cpu())).squeeze().numpy()
    mel_base = db_transform(mel_transform(y_baseline.cpu())).squeeze().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    im1 = axes[0].imshow(mel_true, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title('Ground Truth')
    fig.colorbar(im1, ax=axes[0], format="%+2.0f dB")

    im2 = axes[1].imshow(mel_pred, aspect='auto', origin='lower', cmap='viridis')
    axes[1].set_title('Predicted (Model)')
    fig.colorbar(im2, ax=axes[1], format="%+2.0f dB")
    
    im3 = axes[2].imshow(mel_base, aspect='auto', origin='lower', cmap='viridis')
    axes[2].set_title('Baseline (Interpolation)')
    fig.colorbar(im3, ax=axes[2], format="%+2.0f dB")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def compute_lsd_base10(y_hat, y, n_fft=512):
    """
    Computes LSD using the exact base-10 log formula:
    sqrt( 1/K * sum_k [log10(|X|^2 + 1e-10) - log10(|X_hat|^2 + 1e-10)]^2 )
    """
    window = torch.hann_window(n_fft, device=y.device)
    
    s_hat = torch.stft(y_hat.squeeze(1) if y_hat.ndim > 1 else y_hat, 
                       n_fft, return_complex=True, window=window).abs().pow(2)
    s = torch.stft(y.squeeze(1) if y.ndim > 1 else y, 
                   n_fft, return_complex=True, window=window).abs().pow(2)
    
    log10_s_hat = torch.log10(s_hat + 1e-10)
    log10_s = torch.log10(s + 1e-10)
    
    dist_per_frame = torch.sqrt(torch.mean((log10_s - log10_s_hat)**2, dim=-2))
    lsd = torch.mean(dist_per_frame)
    return lsd.item()

def evaluate_pipeline(model_path, model_name, device, test_low_sr, test_high_sr, visqol_sr):
    print(f"\n{'='*50}")
    print(f"Evaluating {model_name} | {test_low_sr}Hz -> {test_high_sr}Hz")
    print(f"{'='*50}")
    
    # Instantiate the wrapper with predefined dataset statistics
    model = Model(mean=0.0, std=0.0594).to(device)
    model.load_checkpoint(model_path, device)
    model.eval()

    # ViSQOL Setup
    visqol_mode = "audio" if visqol_sr == 48000 else "speech"
    visqol_api = VisqolApi()
    visqol_api.create(mode=visqol_mode)

    metrics = {
        'model_lsd': [], 'base_lsd': [],
        'model_visqol': [], 'base_visqol': []
    }
    
    save_limit = 10
    saved_count = 0
    save_dir_audio = os.path.join('test_outputs', model_name, 'audio')
    save_dir_image = os.path.join('test_outputs', model_name, 'images')
    os.makedirs(save_dir_audio, exist_ok=True)
    os.makedirs(save_dir_image, exist_ok=True)

    with torch.no_grad():
        for idx, (_, hr_wav) in enumerate(tqdm(val100_loader, desc=f"Evaluating")):
            hr_wav = hr_wav.to(device)
            
            # Prepare Ground Truth at Target High SR
            if high_sampling_rate != test_high_sr:
                hr_wav_target = torchaudio.functional.resample(hr_wav, high_sampling_rate, test_high_sr).float()
            else:
                hr_wav_target = hr_wav.float()

            # Prepare Input LR at Target Low SR
            lr_wav_input = torchaudio.functional.resample(hr_wav, high_sampling_rate, test_low_sr).float()

            # 1. Baseline Interpolation
            baseline_hr = torchaudio.functional.resample(lr_wav_input, test_low_sr, test_high_sr)
            
            # 2. Model Prediction (Normalizes input and denormalizes output natively)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                pred = model(lr_wav_input, test_low_sr, test_high_sr)
            
            # Length matching
            min_len = min(pred.shape[-1], hr_wav_target.shape[-1], baseline_hr.shape[-1])
            hr_wav_target = hr_wav_target[..., :min_len]
            pred = pred[..., :min_len].float()
            baseline_hr = baseline_hr[..., :min_len]

            # --- Metrics: LSD ---
            metrics['model_lsd'].append(compute_lsd_base10(pred, hr_wav_target))
            metrics['base_lsd'].append(compute_lsd_base10(baseline_hr, hr_wav_target))

            # --- Metrics: ViSQOL ---
            hr_visqol = torchaudio.functional.resample(hr_wav_target, test_high_sr, visqol_sr).squeeze().cpu().numpy().astype(np.float64)
            pred_visqol = torchaudio.functional.resample(pred, test_high_sr, visqol_sr).squeeze().cpu().numpy().astype(np.float64)
            base_visqol = torchaudio.functional.resample(baseline_hr, test_high_sr, visqol_sr).squeeze().cpu().numpy().astype(np.float64)
            
            try:
                metrics['model_visqol'].append(visqol_api.measure_from_arrays(hr_visqol, pred_visqol, visqol_sr).moslqo)
            except:
                metrics['model_visqol'].append(np.nan)
                
            try:
                metrics['base_visqol'].append(visqol_api.measure_from_arrays(hr_visqol, base_visqol, visqol_sr).moslqo)
            except:
                metrics['base_visqol'].append(np.nan)

            # --- Saving Artifacts ---
            if saved_count < save_limit:
                # Audio
                sf.write(os.path.join(save_dir_audio, f"clip_{idx}_LR.wav"), lr_wav_input.squeeze().cpu().numpy(), test_low_sr)
                sf.write(os.path.join(save_dir_audio, f"clip_{idx}_HR.wav"), hr_wav_target.squeeze().cpu().numpy(), test_high_sr)
                sf.write(os.path.join(save_dir_audio, f"clip_{idx}_Pred.wav"), pred.squeeze().cpu().numpy(), test_high_sr)
                sf.write(os.path.join(save_dir_audio, f"clip_{idx}_Base.wav"), baseline_hr.squeeze().cpu().numpy(), test_high_sr)
                
                # Spectrogram
                img_path = os.path.join(save_dir_image, f"clip_{idx}_specs.png")
                plot_mel_spectrogram(hr_wav_target, pred, baseline_hr, test_high_sr, img_path)
                saved_count += 1

    print(f"\nFinal Results for {model_name}:")
    print(f"Model LSD:       {np.nanmean(metrics['model_lsd']):.4f}")
    print(f"Baseline LSD:    {np.nanmean(metrics['base_lsd']):.4f}")
    print(f"Model ViSQOL:    {np.nanmean(metrics['model_visqol']):.4f}")
    print(f"Baseline ViSQOL: {np.nanmean(metrics['base_visqol']):.4f}")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Explicitly set testing sample rates
    test_low_sr = 24000
    test_high_sr = 48000
    visqol_sr = 48000  # Set to 16000 or 48000

    best_model_path = os.path.join('models', 'lisa_best_model_lsd.pt')
    last_model_path = os.path.join('models', 'lisa_last_model.pt')

    if os.path.exists(best_model_path):
        evaluate_pipeline(best_model_path, 'Best_LSD_Model', device, test_low_sr, test_high_sr, visqol_sr)

    if os.path.exists(last_model_path):
        evaluate_pipeline(last_model_path, 'Last_Epoch_Model', device, test_low_sr, test_high_sr, visqol_sr)

if __name__ == '__main__':
    main()
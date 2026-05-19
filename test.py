import os
import torch
import torchaudio
import soundfile as sf
import matplotlib.pyplot as plt
from torchaudio.functional import resample
from tqdm import tqdm

# Import your custom modules
from data import val_loader, val12_loader
from configs import *
from models import LISA
from extraUtils.loss import log_spectral_distance
from torchmetrics.audio import SignalNoiseRatio

try:
    from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
except ImportError:
    raise ImportError("Please install PESQ support: pip install pesq torchmetrics[audio]")

def plot_mel_spectrogram(y_true, y_pred, sr, save_path):
    """Generates a side-by-side Mel Spectrogram comparison and saves it to disk."""
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

def evaluate_and_save(model_path, model_name, device):
    print(f"\n" + "="*45)
    print(f"--- Evaluating {model_name} ---")
    print("="*45)
    
    # Load Model
    model = LISA().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    snr_metric = SignalNoiseRatio().to(device)
    
    # ---------------------------------------------------------
    # PHASE 1: Standard Validation Set Evaluation (0.5s chunks)
    # ---------------------------------------------------------
    print(f"\n[Phase 1] Evaluating Standard Val Set (Cropped Chunks)...")
    snr_metric.reset()
    avg_lsd_standard = 0.0

    with torch.no_grad():
        for lr_wav, hr_wav in tqdm(val_loader, desc=f"Standard Val"):
            lr_wav, hr_wav = lr_wav.to(device), hr_wav.to(device)
            scale = val_scale
            hsr_new = int(low_sampling_rate * scale)
            hr_wav = resample(hr_wav, high_sampling_rate, hsr_new)
            
            pred = model(lr_wav, scale=scale)
            min_len = min(pred.shape[-1], hr_wav.shape[-1])
            pred, hr_wav = pred[..., :min_len], hr_wav[..., :min_len]
            
            snr_metric(pred, hr_wav)
            avg_lsd_standard += log_spectral_distance(pred, hr_wav).item()

    final_snr_standard = snr_metric.compute().item()
    final_lsd_standard = avg_lsd_standard / len(val_loader)
    
    print(f"\nPhase 1 Results for {model_name}:")
    print(f"Standard Val SNR: {final_snr_standard:.2f} dB")
    print(f"Standard Val LSD: {final_lsd_standard:.4f}")

    # ---------------------------------------------------------
    # PHASE 2: 12 Full-Length Samples Evaluation & Saving
    # ---------------------------------------------------------
    print(f"\n[Phase 2] Evaluating 12 Full-Length Samples & Saving Artifacts...")
    snr_metric.reset()
    pesq_metric = PerceptualEvaluationSpeechQuality(fs=16000, mode='wb').to(device)
    pesq_metric.reset()
    avg_lsd_12 = 0.0
    
    saved_audio_count = 0
    saved_img_count = 0

    with torch.no_grad():
        for idx, (lr_wav, hr_wav) in enumerate(tqdm(val12_loader, desc=f"Full-Length 12")):
            lr_wav, hr_wav = lr_wav.to(device), hr_wav.to(device)
            scale = val_scale
            hsr_new = int(low_sampling_rate * scale)
            hr_wav = resample(hr_wav, high_sampling_rate, hsr_new)
            
            pred = model(lr_wav, scale=scale)
            min_len = min(pred.shape[-1], hr_wav.shape[-1])
            pred, hr_wav = pred[..., :min_len], hr_wav[..., :min_len]
            
            # Normal Metrics
            snr_metric(pred, hr_wav)
            avg_lsd_12 += log_spectral_distance(pred, hr_wav).item()
            
            # PESQ Metric (Must rigorously resample down to 16kHz & remove channel dim)
            hr_16k = resample(hr_wav, hsr_new, 16000).squeeze(1)
            pred_16k = resample(pred, hsr_new, 16000).squeeze(1)
            pesq_metric(pred_16k, hr_16k)

            # --- Artifact Saving Logic ---
            # Save 4 sets of audio clips (LR, SR, and HR ground truth)
            if saved_audio_count < 4:
                lr_save_path = os.path.join('audio', f"{model_name}_sample{saved_audio_count}_LR.wav")
                pred_save_path = os.path.join('audio', f"{model_name}_sample{saved_audio_count}_SR.wav")
                hr_save_path = os.path.join('audio', f"{model_name}_sample{saved_audio_count}_HR.wav")
                
                lr_np = lr_wav[0].cpu().numpy()
                pred_np = pred[0].cpu().numpy()
                hr_np = hr_wav[0].cpu().numpy()
                
                if lr_np.ndim == 2: lr_np = lr_np.T
                if pred_np.ndim == 2: pred_np = pred_np.T
                if hr_np.ndim == 2: hr_np = hr_np.T
                    
                sf.write(lr_save_path, lr_np, low_sampling_rate)
                sf.write(pred_save_path, pred_np, hsr_new)
                sf.write(hr_save_path, hr_np, hsr_new)
                
                saved_audio_count += 1
            
            # Save 2 mel spectrogram comparisons
            elif saved_img_count < 2:
                img_save_path = os.path.join('image', f"{model_name}_mel_comparison_{saved_img_count}.png")
                plot_mel_spectrogram(hr_wav[0], pred[0], hsr_new, img_save_path)
                saved_img_count += 1

    final_snr_12 = snr_metric.compute().item()
    final_lsd_12 = avg_lsd_12 / len(val12_loader)
    final_pesq_12 = pesq_metric.compute().item()
    
    print(f"\nPhase 2 Results for {model_name} (12 Full-Length Samples):")
    print(f"Subset SNR:  {final_snr_12:.2f} dB")
    print(f"Subset LSD:  {final_lsd_12:.4f}")
    print(f"Subset PESQ: {final_pesq_12:.4f} (Wide-Band)")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Ensure directories exist
    os.makedirs('audio', exist_ok=True)
    os.makedirs('image', exist_ok=True)
    
    # Model paths
    best_snr_path = os.path.join('models', 'lisa_best_model_snr.pt')
    best_lsd_path = os.path.join('models', 'lisa_best_model_lsd.pt')

    # Evaluate SNR Model
    if os.path.exists(best_snr_path):
        evaluate_and_save(best_snr_path, 'Best_SNR_Model', device)
    else:
        print(f"Warning: Could not find {best_snr_path}. Skipping.")

    # Evaluate LSD Model
    if os.path.exists(best_lsd_path):
        evaluate_and_save(best_lsd_path, 'Best_LSD_Model', device)
    else:
        print(f"Warning: Could not find {best_lsd_path}. Skipping.")

if __name__ == '__main__':
    main()
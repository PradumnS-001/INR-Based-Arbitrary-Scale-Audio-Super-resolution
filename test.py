import os
import torch
import torchaudio
import soundfile as sf
import matplotlib.pyplot as plt
from torchaudio.functional import resample
from tqdm import tqdm

# Import your custom modules
from data import val_loader
from configs import *
from models import LISA
from extraUtils.loss import log_spectral_distance
from torchmetrics.audio import SignalNoiseRatio

def plot_mel_spectrogram(y_true, y_pred, sr, save_path):
    """
    Generates a side-by-side Mel Spectrogram comparison and saves it to disk.
    """
    # Define transforms
    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_mels=80, n_fft=1024)
    db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)

    # Convert to decibel mel-spectrograms
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
    print(f"\n--- Evaluating {model_name} ---")
    
    # Load Model
    model = LISA().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Initialize Metrics
    snr_metric = SignalNoiseRatio().to(device)
    snr_metric.reset()
    avg_lsd = 0
    
    saved_audio_count = 0
    saved_img_count = 0

    with torch.no_grad():
        for lr_wav, hr_wav in tqdm(val_loader, desc=f"Testing {model_name}"):
            lr_wav, hr_wav = lr_wav.to(device), hr_wav.to(device)
            
            # Setup dynamic target resolution based on val_scale
            scale = val_scale
            hsr_new = int(low_sampling_rate * scale)
            hr_wav = resample(hr_wav, high_sampling_rate, hsr_new)
            
            # Forward Pass
            pred = model(lr_wav, scale=scale)
            
            # Match lengths (in case of fractional padding differences)
            min_len = min(pred.shape[-1], hr_wav.shape[-1])
            pred, hr_wav = pred[..., :min_len], hr_wav[..., :min_len]
            
            # Calculate Metrics
            snr_metric(pred, hr_wav)
            avg_lsd += log_spectral_distance(pred, hr_wav).item()

            # --- Artifact Saving Logic ---
            
            # 1. Save 2 pairs of audio clips
            if saved_audio_count < 2:
                lr_save_path = os.path.join('audio', f"{model_name}_sample{saved_audio_count}_LR.wav")
                pred_save_path = os.path.join('audio', f"{model_name}_sample{saved_audio_count}_SR.wav")
                
                # Convert to numpy
                lr_np = lr_wav[0].cpu().numpy()
                pred_np = pred[0].cpu().numpy()
                
                # soundfile expects shape (frames, channels), PyTorch provides (channels, frames)
                if lr_np.ndim == 2:
                    lr_np = lr_np.T
                if pred_np.ndim == 2:
                    pred_np = pred_np.T
                    
                sf.write(lr_save_path, lr_np, low_sampling_rate)
                sf.write(pred_save_path, pred_np, hsr_new)
                
                saved_audio_count += 1
            
            # 2. Save 1 mel spectrogram comparison (use the 3rd sample we encounter)
            elif saved_img_count < 1:
                img_save_path = os.path.join('image', f"{model_name}_mel_comparison.png")
                plot_mel_spectrogram(hr_wav[0], pred[0], hsr_new, img_save_path)
                saved_img_count += 1

    # Final Compute
    final_snr = snr_metric.compute().item()
    final_lsd = avg_lsd / len(val_loader)
    
    print(f"\nFinal Results for {model_name}:")
    print(f"Overall Test SNR: {final_snr:.2f} dB")
    print(f"Overall Test LSD: {final_lsd:.4f}")

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
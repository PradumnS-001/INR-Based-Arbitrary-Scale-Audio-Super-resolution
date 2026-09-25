import os
import math
import torch
import torchaudio
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from torchaudio.functional import resample
from tqdm import tqdm

# Handle Visqol import gracefully
try:
    from visqol import VisqolApi
    HAS_VISQOL = True
except ImportError:
    HAS_VISQOL = False

# Import your custom modules
# FIX: Explicitly importing the 200 clips loader
from data import val200_loader
from configs import *
from models import LISA

# ==============================================================================
# 1. Metric Computation Functions
# ==============================================================================

def log_spectral_distance(y_hat:torch.Tensor, y:torch.Tensor, n_fft = 512) -> float:
    """
    Measures the log spectral distance (LSD) in acoustic Decibels (dB).
    """
    window = torch.hann_window(n_fft, device=y.device)
        
    s_hat = torch.stft(y_hat.squeeze(1) if y_hat.ndim > 1 else y_hat, 
                        n_fft, return_complex=True, window=window).abs().pow(2)
    s = torch.stft(y.squeeze(1) if y.ndim > 1 else y, 
                    n_fft, return_complex=True, window=window).abs().pow(2)
    
    # 1e-7 establishes a strict -70dB physical noise floor 
    log10_s_hat = torch.log10(s_hat + 1e-7)
    log10_s = torch.log10(s + 1e-7)
    
    dist_per_frame = torch.sqrt(torch.mean((log10_s - log10_s_hat)**2, dim=-2))
    
    # 10.0 multiplier converts Bels to acoustic dB
    return torch.mean(dist_per_frame).item()


class ViSQOLManager:
    """Manages separate 16kHz (Speech) and 48kHz (Audio) ViSQOL sessions."""
    def __init__(self):
        self.api_16k = None
        self.api_48k = None
        if HAS_VISQOL:
            try:
                self.api_16k = VisqolApi()
                self.api_16k.create(mode="speech")
                self.api_48k = VisqolApi()
                self.api_48k.create(mode="audio")
            except Exception as e:
                print(f"[Warning] Failed to initialize ViSQOL: {e}")

    def measure(self, ref_wav: np.ndarray, deg_wav: np.ndarray, target_sr: int) -> float:
        if not HAS_VISQOL:
            return float('nan')
        try:
            if target_sr == 16000 and self.api_16k is not None:
                return self.api_16k.measure_from_arrays(ref_wav, deg_wav, 16000).moslqo
            elif target_sr == 48000 and self.api_48k is not None:
                return self.api_48k.measure_from_arrays(ref_wav, deg_wav, 48000).moslqo
            else:
                return float('nan')
        except Exception:
            return float('nan')


def plot_mel_spectrogram(y_true, y_pred, sr, save_path):
    """Generates a side-by-side Mel Spectrogram comparison and saves it to disk."""
    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_mels=80, n_fft=1024)
    db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)

    mel_true = db_transform(mel_transform(y_true.cpu())).squeeze().numpy()
    mel_pred = db_transform(mel_transform(y_pred.cpu())).squeeze().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = axes[0].imshow(mel_true, aspect='auto', origin='lower', cmap='magma')
    axes[0].set_title('Ground Truth Mel Spectrogram')
    axes[0].set_ylabel('Mel bins')
    axes[0].set_xlabel('Frames')
    fig.colorbar(im1, ax=axes[0], format="%+2.0f dB")

    im2 = axes[1].imshow(mel_pred, aspect='auto', origin='lower', cmap='magma')
    axes[1].set_title('Predicted (LISA) Mel Spectrogram')
    axes[1].set_xlabel('Frames')
    fig.colorbar(im2, ax=axes[1], format="%+2.0f dB")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

# ==============================================================================
# 2. Main Evaluation Pipeline
# ==============================================================================

def evaluate_and_save(model_path, model_name, device, test_low_sr, test_high_sr):
    print(f"\n" + "="*50)
    print(f"--- Evaluating {model_name} ---")
    print(f"--- Pair: {test_low_sr} Hz -> {test_high_sr} Hz ---")
    print("="*50)
    
    # Load Model
    model = LISA().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    visqol_mgr = ViSQOLManager()

    # ---------------------------------------------------------
    # Exact Continuous Scale Calculation (Approximator Removed)
    # ---------------------------------------------------------
    calc_high_sr = test_high_sr
    true_scale = test_high_sr / test_low_sr

    # ---------------------------------------------------------
    # Evaluation Loop
    # ---------------------------------------------------------
    lsd_scores = []
    visqol_scores = []
    
    saved_count = 0
    save_limit = 20

    with torch.no_grad():
        # FIX: Explicitly iterating over val200_loader
        for idx, (_, hr_wav) in enumerate(tqdm(val200_loader, desc=f"Evaluating Clips")):
            hr_wav = hr_wav.to(device)
            
            # Prepare exact target and input using torchaudio resample
            hr_target = resample(hr_wav, high_sampling_rate, calc_high_sr)
            lr_input = resample(hr_wav, high_sampling_rate, test_low_sr)
            
            # Model prediction in continuous float space
            pred_raw = model(lr_input, scale=true_scale)
            
            min_len = min(pred_raw.shape[-1], hr_target.shape[-1])
            pred_cut = pred_raw[..., :min_len]
            hr_cut = hr_target[..., :min_len]
            
            # 1. LSD (Unclamped for pristine spectral accuracy)
            lsd_scores.append(log_spectral_distance(pred_cut, hr_cut))
            
            # 2. ViSQOL (Strictly Clamped to prevent DSP errors)
            if calc_high_sr in [16000, 48000] and HAS_VISQOL:
                pred_clamped = torch.clamp(pred_cut, -0.999, 0.999)
                ref_np = hr_cut.squeeze().detach().cpu().numpy().astype(np.float64)
                deg_np = pred_clamped.squeeze().detach().cpu().numpy().astype(np.float64)
                visqol_scores.append(visqol_mgr.measure(ref_np, deg_np, calc_high_sr))

            # 3. Save exactly 20 Spectrograms and Audio Artifacts
            if saved_count < save_limit:
                # Plot Spectrogram
                img_save_path = os.path.join('image1', f"{model_name}_mel_comparison_{idx}.png")
                plot_mel_spectrogram(hr_cut[0], pred_cut[0], calc_high_sr, img_save_path)
                
                # Save Audio (Clamped)
                pred_clamped = torch.clamp(pred_cut, -0.999, 0.999)
                
                # Slicing the first batch item [0] to prevent multi-channel dimension errors in soundfile
                lr_np = lr_input[0].squeeze().detach().cpu().numpy().astype(np.float32)
                pred_np = pred_clamped[0].squeeze().detach().cpu().numpy().astype(np.float32)
                hr_np = hr_cut[0].squeeze().detach().cpu().numpy().astype(np.float32)
                
                sf.write(os.path.join('audio1', f"{model_name}_sample{idx}_LR_{test_low_sr}.wav"), lr_np, test_low_sr)
                sf.write(os.path.join('audio1', f"{model_name}_sample{idx}_SR_{calc_high_sr}.wav"), pred_np, calc_high_sr)
                sf.write(os.path.join('audio1', f"{model_name}_sample{idx}_HR_{calc_high_sr}.wav"), hr_np, calc_high_sr)
                
                saved_count += 1

    final_lsd = float(np.nanmean(lsd_scores))
    final_visqol = float(np.nanmean(visqol_scores)) if len(visqol_scores) > 0 else float('nan')
    
    print(f"\nFinal Results for {model_name} ({test_low_sr} Hz -> {calc_high_sr} Hz):")
    print(f"Mean LSD:    {final_lsd:.4f} dB")
    if not math.isnan(final_visqol):
        print(f"Mean ViSQOL: {final_visqol:.4f} MOS-LQO")
    else:
        print("Mean ViSQOL: N/A (Only supports 16kHz or 48kHz target)")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Ensure directories exist
    os.makedirs('audio1', exist_ok=True)
    os.makedirs('image1', exist_ok=True)
    
    # --- Set Specific Evaluation Pair Here ---
    target_input_sr = 16000
    target_output_sr = 48000
    
    best_lsd_path = os.path.join('models', 'lisa_final_epoch01.pt')

    # Evaluate Model
    if os.path.exists(best_lsd_path):
        evaluate_and_save(best_lsd_path, 'Best_LSD_Model', device, target_input_sr, target_output_sr)
    else:
        print(f"Warning: Could not find {best_lsd_path}.")

if __name__ == '__main__':
    main()
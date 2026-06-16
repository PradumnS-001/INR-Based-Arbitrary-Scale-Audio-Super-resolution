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
from models import ImprovedLISA
from extraUtils.loss import log_spectral_distance, compute_audio_ssim
from extraUtils.misc import count_params

try:
    from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
except ImportError:
    raise ImportError("Please install PESQ support: pip install pesq torchmetrics[audio]")

def plot_mel_spectrogram_triple(y_true, y_det, y_stoc, sr, save_path):
    """Generates a 3-way Mel Spectrogram comparison (Truth vs Det vs Stoc)."""
    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_mels=80, n_fft=512)
    db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)

    mel_true = db_transform(mel_transform(y_true.cpu())).squeeze().numpy()
    mel_det = db_transform(mel_transform(y_det.cpu())).squeeze().numpy()
    mel_stoc = db_transform(mel_transform(y_stoc.cpu())).squeeze().numpy()

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    im0 = axes[0].imshow(mel_true, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title('Ground Truth')
    fig.colorbar(im0, ax=axes[0], format="%+2.0f dB")

    im1 = axes[1].imshow(mel_det, aspect='auto', origin='lower', cmap='viridis')
    axes[1].set_title('Deterministic (Regression to Mean)')
    fig.colorbar(im1, ax=axes[1], format="%+2.0f dB")

    im2 = axes[2].imshow(mel_stoc, aspect='auto', origin='lower', cmap='viridis')
    axes[2].set_title('Stochastic (Gamma Noise Added)')
    fig.colorbar(im2, ax=axes[2], format="%+2.0f dB")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def evaluate_and_save(model_path, model_name, device):
    print(f"\n" + "="*60)
    print(f"--- {model_name} ---")
    print("="*60)
    
    # ---------------------------------------------------------
    # PHASE 1: Load the Model
    # ---------------------------------------------------------
    print(f"[Phase 1] Loading Model from {model_path}...")
    model = ImprovedLISA().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # ---------------------------------------------------------
    # PHASE 3: Mathematical Metrics (LSD & SSIM)
    # ---------------------------------------------------------
    print(f"\n[Phase 3] Computing Structural Metrics (Stochastic)...")
    torch.manual_seed(42) 
    
    avg_lsd, avg_ssim = 0.0, 0.0

    with torch.no_grad():
        for lr_wav, hr_wav in tqdm(val_loader, desc=f"Metrics Eval"):
            lr_wav, hr_wav = lr_wav.to(device), hr_wav.to(device)
            scale = val_scale
            hsr_new = int(low_sampling_rate * scale)
            
            hr_wav = resample(hr_wav, high_sampling_rate, hsr_new)
            lr_wave_base = resample(lr_wav, low_sampling_rate, hsr_new)
            
            # STOCHASTIC INFERENCE
            pred, _ = model(lr_wav, scale=scale, infer_stoc=True)
            
            min_len = min(pred.shape[-1], hr_wav.shape[-1])
            pred, hr_wav = pred[..., :min_len], hr_wav[..., :min_len]
            pred += lr_wave_base[..., :min_len]
            
            avg_lsd += log_spectral_distance(pred, hr_wav).item()
            avg_ssim += compute_audio_ssim(pred, hr_wav, sample_rate=hsr_new)

    num_batches = len(val_loader)
    print(f"\nPhase 3 Results for {model_name}:")
    print(f"--> SSIM:             {avg_ssim / num_batches:.4f}")
    print(f"--> LSD:              {avg_lsd / num_batches:.4f}")

    # ---------------------------------------------------------
    # PHASE 4: Full-Clip Artifacts & PESQ Eval
    # ---------------------------------------------------------
    print(f"\n[Phase 4] Evaluating PESQ & Generating Dual Artifacts...")
    
    pesq_metric = PerceptualEvaluationSpeechQuality(fs=16000, mode='wb').to(device)
    avg_pesq_det, avg_pesq_stoc = 0.0, 0.0
    pesq_count = 0

    with torch.no_grad():
        for idx, (lr_wav, hr_wav) in enumerate(tqdm(val12_loader, desc=f"Artifacts & PESQ")):
            if idx >= 12: break

            lr_wav, hr_wav = lr_wav.to(device), hr_wav.to(device)
            scale = val_scale
            hsr_new = int(low_sampling_rate * scale)
            
            hr_wav = resample(hr_wav, high_sampling_rate, hsr_new)
            lr_wave_base = resample(lr_wav, low_sampling_rate, hsr_new)
            
            # INFERENCE PASSES
            pred_det, _ = model(lr_wav, scale=scale, infer_stoc=False)
            pred_stoc, _ = model(lr_wav, scale=scale, infer_stoc=True)
            
            min_len = min(pred_det.shape[-1], hr_wav.shape[-1])
            hr_wav = hr_wav[..., :min_len]
            base_add = lr_wave_base[..., :min_len]
            
            pred_det = pred_det[..., :min_len] + base_add
            pred_stoc = pred_stoc[..., :min_len] + base_add

            # --- PESQ Calculation (Needs 16kHz) ---
            pred_det_16k = resample(pred_det, hsr_new, 16000).squeeze(1)
            pred_stoc_16k = resample(pred_stoc, hsr_new, 16000).squeeze(1)
            hr_16k = resample(hr_wav, hsr_new, 16000).squeeze(1)

            try:
                avg_pesq_det += pesq_metric(pred_det_16k, hr_16k).item()
                avg_pesq_stoc += pesq_metric(pred_stoc_16k, hr_16k).item()
                pesq_count += 1
            except Exception:
                pass # Silently skip if a clip still fails PESQ's utterance test

            # --- Save Audio (Indices 0 to 7) ---
            if idx < 8:
                sf.write(os.path.join('audio', f"{model_name}_{idx}_LR.wav"), lr_wav[0].cpu().numpy().T, low_sampling_rate)
                sf.write(os.path.join('audio', f"{model_name}_{idx}_HR.wav"), hr_wav[0].cpu().numpy().T, hsr_new)
                
                det_out = pred_det[0].cpu().numpy().T
                det_out /= max(abs(det_out).max(), 1e-5)
                sf.write(os.path.join('audio', f"{model_name}_{idx}_SR_Det.wav"), det_out, hsr_new)
                
                stoc_out = pred_stoc[0].cpu().numpy().T
                stoc_out /= max(abs(stoc_out).max(), 1e-5)
                sf.write(os.path.join('audio', f"{model_name}_{idx}_SR_Stoc.wav"), stoc_out, hsr_new)
            
            # --- Save Spectrograms (Indices 8 to 11) ---
            elif idx >= 8 and idx < 12:
                img_save_path = os.path.join('image', f"{model_name}_mel_comparison_{idx}.png")
                plot_mel_spectrogram_triple(hr_wav[0], pred_det[0], pred_stoc[0], hsr_new, img_save_path)

    if pesq_count > 0:
        print(f"\nPhase 4 PESQ Results for {model_name} ({pesq_count} clips):")
        print(f"--> Deterministic PESQ: {avg_pesq_det / pesq_count:.4f}")
        print(f"--> Stochastic PESQ:    {avg_pesq_stoc / pesq_count:.4f}")
    else:
        print("\nPhase 4 PESQ Results: Failed to compute (no valid utterances).")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    os.makedirs('audio', exist_ok=True)
    os.makedirs('image', exist_ok=True)
    
    best_ssim_path = os.path.join('models', 'lisa_best_model_ssim.pth')
    best_lsd_path = os.path.join('models', 'lisa_best_model_lsd.pth')

    if os.path.exists(best_ssim_path): 
        evaluate_and_save(best_ssim_path, 'Best_SSIM_Model', device)
    else:
        print(f"Warning: Could not find {best_ssim_path}. Skipping.")

    if os.path.exists(best_lsd_path): 
        evaluate_and_save(best_lsd_path, 'Best_LSD_Model', device)
    else:
        print(f"Warning: Could not find {best_lsd_path}. Skipping.")

if __name__ == '__main__':
    model = ImprovedLISA()
    count_params(model)
    main()
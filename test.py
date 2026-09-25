import os
import math
import torch
import torchaudio
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Handle Visqol import gracefully if not compiled in current environment
try:
    from visqol import VisqolApi
    HAS_VISQOL = True
except ImportError:
    HAS_VISQOL = False

from configs import *
from data import test_loader
from models import SIRIUS
from utility import count_params

# ==============================================================================
# 1. Resampling & Audio Caching Helpers
# ==============================================================================

class AudioResampleCache:
    """Caches torchaudio Resample instances to avoid rebuilding sinc filters."""
    def __init__(self, device):
        self.device = device
        self._cache = {}

    def resample(self, wav: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
        if orig_sr == target_sr:
            return wav
        key = (orig_sr, target_sr)
        if key not in self._cache:
            self._cache[key] = torchaudio.transforms.Resample(
                orig_freq=orig_sr, new_freq=target_sr
            ).to(self.device)
        return self._cache[key](wav)


def preload_validation_set(loader, device):
    """Preloads validation audio into memory to prevent repeated disk I/O."""
    print("Pre-loading validation audio dataset into memory...")
    cached_hr = []
    for hr_wav in loader:
        cached_hr.append(hr_wav.to(device).float())
    return cached_hr

# ==============================================================================
# 2. Metric Computation Functions
# ==============================================================================
from functions import log_spectral_distance

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

# ==============================================================================
# 3. Formatted Visualizations (Spectrograms & Comparison Sweeps)
# ==============================================================================

def plot_unified_spectrograms(gt_48k, pred_8_16, pred_8_48, pred_16_48, save_path):
    """Renders a 1x4 grid containing Ground Truth and the 3 target predictions."""
    panels = [
        ("Ground Truth (48 kHz)", gt_48k.cpu(), 48000),
        ("Prediction: 8 kHz → 16 kHz", pred_8_16.cpu(), 16000),
        ("Prediction: 8 kHz → 48 kHz", pred_8_48.cpu(), 48000),
        ("Prediction: 16 kHz → 48 kHz", pred_16_48.cpu(), 48000),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(40, 10))
    axes = axes.flatten()

    for idx, (title, wav, sr) in enumerate(panels):
        n_fft = 2048 if sr >= 32000 else 1024
        hop = n_fft // 4
        mel_tf = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_mels=128, n_fft=n_fft, hop_length=hop)
        db_tf = torchaudio.transforms.AmplitudeToDB(top_db=80)

        # Added .detach() for safety
        mel_spec = db_tf(mel_tf(wav)).squeeze().detach().cpu().numpy()
        duration_sec = wav.shape[-1] / sr
        freq_max_khz = (sr / 2.0) / 1000.0

        im = axes[idx].imshow(
            mel_spec,
            aspect='auto',
            origin='lower',
            cmap='magma',
            extent=[0, duration_sec, 0, freq_max_khz]
        )
        axes[idx].set_title(title, fontsize=13, fontweight='bold', pad=8)
        axes[idx].set_xlabel("Time (s)", fontsize=11)
        axes[idx].set_ylabel("Frequency (kHz)", fontsize=11)
        cbar = fig.colorbar(im, ax=axes[idx], format="%+2.0f dB")
        cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close()

def plot_scale_sweep(scales, lsd_values, title, xlabel, save_path):
    """Generates a clean, single-axis publication-ready plot strictly for LSD vs scale."""
    fig, ax1 = plt.subplots(figsize=(10, 5.5))

    color_lsd = '#d62728'
    ax1.set_xlabel(xlabel, fontsize=12, fontweight='bold', labelpad=8)
    ax1.set_ylabel("Log Spectral Distance (LSD) [dB] ↓", color=color_lsd, fontsize=12, fontweight='bold')
    
    valid_lsd = [(s, v) for s, v in zip(scales, lsd_values) if not math.isnan(v)]
    if valid_lsd:
        s_lsd, v_lsd = zip(*valid_lsd)
        ax1.plot(s_lsd, v_lsd, color=color_lsd, marker='o', linewidth=2.2, label='LSD (Lower is better)')

    ax1.tick_params(axis='y', labelcolor=color_lsd, labelsize=11)
    ax1.tick_params(axis='x', labelsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax1.legend(loc='best', framealpha=0.9, fontsize=10)
    plt.title(title, fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()

# ==============================================================================
# 4. Core Pipeline Evaluation Routine
# ==============================================================================

@torch.no_grad()
def evaluate_pair(model, cached_hr, resampler, low_sr, high_sr, visqol_mgr, compute_visqol=True):
    """Evaluates the model over a single (low_sr -> high_sr) pair across all clips."""
    lsd_scores = []
    base_lsd_scores = []
    visqol_scores = []
    base_visqol_scores = []

    for hr_wav in cached_hr:
        hr_target = resampler.resample(hr_wav, high_sampling_rate, high_sr)
        lr_input = resampler.resample(hr_wav, high_sampling_rate, low_sr)
        baseline_hr = resampler.resample(lr_input, low_sr, high_sr)

        pred_raw = model(lr_input, low_sr, high_sr)

        min_len = min(pred_raw.shape[-1], hr_target.shape[-1], baseline_hr.shape[-1])
        pred_cut = pred_raw[..., :min_len]
        target_cut = hr_target[..., :min_len]
        base_cut = baseline_hr[..., :min_len]

        # UNCLAMPED evaluation for pristine spectral gradients
        lsd_scores.append(log_spectral_distance(pred_cut, target_cut).detach().item())
        base_lsd_scores.append(log_spectral_distance(base_cut, target_cut).detach().item())

        # CLAMPED evaluation for DSP constraints
        if compute_visqol and high_sr in (16000, 48000) and HAS_VISQOL:
            pred_clamped = torch.clamp(pred_cut, -0.999, 0.999)
            base_clamped = torch.clamp(base_cut, -0.999, 0.999)
            
            # Explicitly detached before converting to numpy
            ref_np = target_cut.squeeze().detach().cpu().numpy().astype(np.float64)
            deg_np = pred_clamped.squeeze().detach().cpu().numpy().astype(np.float64)
            base_np = base_clamped.squeeze().detach().cpu().numpy().astype(np.float64)
            
            visqol_scores.append(visqol_mgr.measure(ref_np, deg_np, high_sr))
            base_visqol_scores.append(visqol_mgr.measure(ref_np, base_np, high_sr))

    mean_lsd = float(np.nanmean(lsd_scores))
    mean_base_lsd = float(np.nanmean(base_lsd_scores))
    mean_visqol = float(np.nanmean(visqol_scores)) if len(visqol_scores) > 0 else float('nan')
    mean_base_visqol = float(np.nanmean(base_visqol_scores)) if len(base_visqol_scores) > 0 else float('nan')
    
    return mean_lsd, mean_base_lsd, mean_visqol, mean_base_visqol


def print_ascii_table(title: str, headers: list[str], rows: list[list[str]]):
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    sep = "+-" + "-+-".join(["-" * w for w in col_widths]) + "-+"
    header_str = "| " + " | ".join([f"{h:<{w}}" for h, w in zip(headers, col_widths)]) + " |"

    print(f"\n{title}")
    print(sep)
    print(header_str)
    print(sep)
    for row in rows:
        row_str = "| " + " | ".join([f"{str(v):<{w}}" for v, w in zip(row, col_widths)]) + " |"
        print(row_str)
    print(sep)

# ==============================================================================
# 5. Master Execution Function
# ==============================================================================

def run_comprehensive_evaluation(
    model_path: str,
    model_name: str,
    device: str,
    custom_low_sr: int = 11025,
    custom_high_sr: int = 44100
):
    print(f"\n{'='*70}\nSTARTING COMPREHENSIVE EVALUATION FOR: {model_name}\n{'='*70}")

    out_dir_audio = os.path.join('test_outputs', model_name, 'audio')
    out_dir_img = os.path.join('test_outputs', model_name, 'images')
    out_dir_plots = os.path.join('test_outputs', model_name, 'plots')
    os.makedirs(out_dir_audio, exist_ok=True)
    os.makedirs(out_dir_img, exist_ok=True)
    os.makedirs(out_dir_plots, exist_ok=True)

    model = SIRIUS(mean=0.0, std=0.0594).to(device)
    count_params(model=model)
    model.load_checkpoint(model_path, device)
    model.eval()

    resampler = AudioResampleCache(device)
    visqol_mgr = ViSQOLManager()
    cached_hr = preload_validation_set(test_loader, device)

    # --------------------------------------------------------------------------
    # Task A: Multi-Scale LSD Evaluation
    # --------------------------------------------------------------------------
    lsd_pairs = [
        (8000, 16000), (8000, 24000), (8000, 48000),
        (16000, 24000), (16000, 48000),
        (24000, 48000),
        (12000, 24000), (12000, 48000),
        (custom_low_sr, custom_high_sr)
    ]

    print("\n--- Running Multi-Scale LSD Suite ---")
    lsd_table_rows = []
    for (l_sr, h_sr) in tqdm(lsd_pairs, desc="Evaluating LSD Pairs"):
        scale_ratio = h_sr / l_sr
        tag = "Custom" if (l_sr, h_sr) == (custom_low_sr, custom_high_sr) else "Standard"
        lsd_val, base_lsd, _, _ = evaluate_pair(model, cached_hr, resampler, l_sr, h_sr, visqol_mgr, compute_visqol=False)
        lsd_table_rows.append([f"{l_sr} Hz", f"{h_sr} Hz", f"{scale_ratio:.2f}x", tag, f"{lsd_val:.4f}", f"{base_lsd:.4f}"])

    print_ascii_table(
        f"LSD Multi-Scale Evaluation ({model_name})",
        ["Input SR", "Target SR", "Scale", "Type", "Model LSD [dB]", "Base LSD [dB]"],
        lsd_table_rows
    )

    # --------------------------------------------------------------------------
    # Task B: Upward ViSQOL Evaluation
    # --------------------------------------------------------------------------
    visqol_pairs = [
        (8000, 16000), (12000, 16000),
        (8000, 48000), (12000, 48000), (16000, 48000), (24000, 48000)
    ]

    print("\n--- Running Upward ViSQOL Suite ---")
    visqol_table_rows = []
    for (l_sr, h_sr) in tqdm(visqol_pairs, desc="Evaluating ViSQOL Pairs"):
        mode = "Speech (16k)" if h_sr == 16000 else "Audio (48k)"
        _, _, visqol_val, base_visqol_val = evaluate_pair(model, cached_hr, resampler, l_sr, h_sr, visqol_mgr, compute_visqol=True)
        
        vis_str = f"{visqol_val:.4f}" if not math.isnan(visqol_val) else "N/A"
        base_vis_str = f"{base_visqol_val:.4f}" if not math.isnan(base_visqol_val) else "N/A"
        
        visqol_table_rows.append([f"{l_sr} Hz", f"{h_sr} Hz", mode, vis_str, base_vis_str])

    print_ascii_table(
        f"ViSQOL Upward Evaluation ({model_name})",
        ["Input SR", "Target Anchor", "ViSQOL Mode", "Model MOS", "Base MOS"],
        visqol_table_rows
    )

    # --------------------------------------------------------------------------
    # Task C: Mel-Spectrogram & Audio Generation
    # --------------------------------------------------------------------------
    print("\n--- Saving Spectrograms and Audio Artifacts ---")
    num_to_save = min(20, len(cached_hr))

    for idx in range(num_to_save):
        hr_raw = cached_hr[idx]
        gt_48k = resampler.resample(hr_raw, high_sampling_rate, 48000)

        lr_8k = resampler.resample(hr_raw, high_sampling_rate, 8000)
        lr_16k = resampler.resample(hr_raw, high_sampling_rate, 16000)

        pred_8_16 = torch.clamp(model(lr_8k, 8000, 16000), -0.999, 0.999)
        pred_8_48 = torch.clamp(model(lr_8k, 8000, 48000), -0.999, 0.999)
        pred_16_48 = torch.clamp(model(lr_16k, 16000, 48000), -0.999, 0.999)

        spec_path = os.path.join(out_dir_img, f"clip_{idx}_unified_mel.png")
        plot_unified_spectrograms(gt_48k, pred_8_16, pred_8_48, pred_16_48, spec_path)

        sf.write(os.path.join(out_dir_audio, f"clip_{idx}_GT_48k.wav"), gt_48k.squeeze().detach().cpu().numpy(), 48000)
        sf.write(os.path.join(out_dir_audio, f"clip_{idx}_Pred_8k_to_16k.wav"), pred_8_16.squeeze().detach().cpu().numpy(), 16000)
        sf.write(os.path.join(out_dir_audio, f"clip_{idx}_Pred_8k_to_48k.wav"), pred_8_48.squeeze().detach().cpu().numpy(), 48000)
        sf.write(os.path.join(out_dir_audio, f"clip_{idx}_Pred_16k_to_48k.wav"), pred_16_48.squeeze().detach().cpu().numpy(), 48000)

# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    custom_low_sr = 11025
    custom_high_sr = 44100

    last_model_path = os.path.join('models', 'sirius_last_model.pt')

    if os.path.exists(last_model_path):
        run_comprehensive_evaluation(
            model_path=last_model_path,
            model_name='Last_Epoch_Model',
            device=device,
            custom_low_sr=custom_low_sr,
            custom_high_sr=custom_high_sr
        )
    else:
        print(f"Error: No checkpoints found at '{last_model_path}'.")

if __name__ == '__main__':
    main()
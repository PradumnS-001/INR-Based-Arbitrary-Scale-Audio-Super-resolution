import io
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st
import torch
import torchaudio

from models import Model


AUDIO_EXTENSIONS = {".wav", ".flac"}
DEFAULT_MEAN = 0.0
DEFAULT_STD = 0.0594
CANONICAL_HIGH_SR = 48000


st.set_page_config(
    page_title="Audio Super-Resolution Dashboard",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def scan_sample_audio(base_dir: str):
    root = Path(base_dir)
    if not root.exists():
        return []

    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        try:
            info = sf.info(path.as_posix())
        except Exception:
            continue
        if info.samplerate == CANONICAL_HIGH_SR:
            files.append(str(path))
    return files


@st.cache_resource
def load_model(checkpoint_path: str, device: str):
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = Model(mean=DEFAULT_MEAN, std=DEFAULT_STD).to(device)
    try:
        model.load_checkpoint(str(checkpoint), device)
    except Exception as exc:
        raise RuntimeError(f"Failed to load checkpoint '{checkpoint.name}': {exc}") from exc

    model.eval()
    return model


def to_mono_if_needed(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.ndim == 1:
        return waveform.unsqueeze(0)
    if waveform.ndim == 2:
        if waveform.shape[0] > 1:
            return waveform.mean(dim=0, keepdim=True)
        return waveform
    if waveform.ndim == 3:
        if waveform.shape[0] > 1:
            return waveform.mean(dim=0, keepdim=True)
        return waveform
    raise ValueError(f"Unsupported waveform shape: {tuple(waveform.shape)}")


def normalize_waveform(waveform: torch.Tensor) -> torch.Tensor:
    waveform = waveform.float()
    waveform = waveform.clamp(-1.0, 1.0)
    return waveform


def load_audio_from_path(path: str):
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 1:
        waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    else:
        waveform = torch.tensor(audio.T, dtype=torch.float32)
    waveform = normalize_waveform(to_mono_if_needed(waveform))
    return waveform, sample_rate


def load_audio_from_uploaded(file_obj):
    wav_bytes = file_obj.read()
    if not wav_bytes:
        raise ValueError("Uploaded audio is empty.")

    buffer = io.BytesIO(wav_bytes)
    audio, sample_rate = sf.read(buffer, dtype="float32", always_2d=False)
    if sample_rate != CANONICAL_HIGH_SR:
        raise ValueError(
            f"Only 48 kHz ground-truth audio is accepted for this project. "
            f"This file is {sample_rate} Hz; resample it to 48 kHz before uploading."
        )
    if audio.ndim == 1:
        waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    else:
        waveform = torch.tensor(audio.T, dtype=torch.float32)
    waveform = normalize_waveform(to_mono_if_needed(waveform))
    return waveform, sample_rate


def human_readable_sr(value: int) -> str:
    return f"{value:,} Hz"


def make_wav_bytes(waveform: torch.Tensor, sample_rate: int) -> bytes:
    waveform = waveform.detach().cpu().float().squeeze()
    if waveform.ndim == 0:
        waveform = waveform.unsqueeze(0)

    buffer = io.BytesIO()
    sf.write(buffer, waveform.numpy(), sample_rate, format="WAV")
    buffer.seek(0)
    return buffer.getvalue()


def plot_spectrogram(waveform: torch.Tensor, sample_rate: int, title: str):
    waveform = waveform.detach().cpu().float().squeeze()
    if waveform.ndim == 0:
        waveform = waveform.unsqueeze(0)

    fig, ax = plt.subplots(figsize=(10, 3.6))
    spec = torchaudio.transforms.Spectrogram(
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        power=2,
    )(waveform)
    spec_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)(spec)
    spec_db = spec_db.squeeze(0).cpu().numpy()

    img = ax.imshow(
        spec_db,
        aspect="auto",
        origin="lower",
        cmap="magma",
        extent=[0, waveform.shape[-1] / sample_rate, 0, sample_rate / 2],
    )
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    return fig


def compute_lsd_base10(y_hat: torch.Tensor, y: torch.Tensor, n_fft: int = 512):
    y_hat = y_hat.detach().float().squeeze()
    y = y.detach().float().squeeze()

    if y_hat.ndim == 0:
        y_hat = y_hat.unsqueeze(0)
    if y.ndim == 0:
        y = y.unsqueeze(0)

    window = torch.hann_window(n_fft, device=y.device)
    s_hat = torch.stft(
        y_hat.unsqueeze(0) if y_hat.ndim == 1 else y_hat,
        n_fft,
        return_complex=True,
        window=window,
    ).abs().pow(2)
    s = torch.stft(
        y.unsqueeze(0) if y.ndim == 1 else y,
        n_fft,
        return_complex=True,
        window=window,
    ).abs().pow(2)

    log10_s_hat = torch.log10(s_hat + 1e-10)
    log10_s = torch.log10(s + 1e-10)
    dist_per_frame = torch.sqrt(torch.mean((log10_s - log10_s_hat) ** 2, dim=-2))
    return float(torch.mean(dist_per_frame).item())


def compute_visqol_mos(ref: torch.Tensor, hyp: torch.Tensor, sample_rate: int):
    try:
        from visqol import VisqolApi
    except Exception:
        return None

    try:
        ref_np = ref.detach().cpu().numpy().astype(np.float64)
        hyp_np = hyp.detach().cpu().numpy().astype(np.float64)
        mode = "audio" if sample_rate >= 16000 else "speech"
        api = VisqolApi()
        api.create(mode=mode)
        return float(api.measure_from_arrays(ref_np, hyp_np, sample_rate).moslqo)
    except Exception:
        return None


def build_audio_outputs(gt_waveform: torch.Tensor, gt_sr: int, low_sr: int, high_sr: int, model, device: str):
    if gt_waveform.shape[-1] < max(256, int(0.1 * gt_sr)):
        raise ValueError(
            f"Selected audio is too short. Use at least {max(256, int(0.1 * gt_sr))} samples ({0.1:.2f} s) at {gt_sr} Hz."
        )

    # Repo rule: the model evaluates against a canonical 48 kHz high-resolution reference.
    # If the native file is already at 48 kHz, no extra resample occurs. Otherwise it is resampled
    # to 48 kHz first, then we downsample to the requested low-res input and resample back to the task high SR.
    gt_reference = gt_waveform.to(device)
    if gt_sr != CANONICAL_HIGH_SR:
        gt_reference = torchaudio.functional.resample(gt_reference, gt_sr, CANONICAL_HIGH_SR).to(device)

    low_input = torchaudio.functional.resample(gt_reference, CANONICAL_HIGH_SR, low_sr).float().to(device)
    baseline = torchaudio.functional.resample(low_input, low_sr, high_sr).to(device)

    model_input = low_input.unsqueeze(1)
    with torch.no_grad():
        if device.startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model(model_input, low_sr, high_sr)
        else:
            pred = model(model_input, low_sr, high_sr)

    gt_target = gt_reference
    if high_sr != CANONICAL_HIGH_SR:
        gt_target = torchaudio.functional.resample(gt_reference, CANONICAL_HIGH_SR, high_sr).to(device)

    min_len = min(gt_target.shape[-1], pred.shape[-1], baseline.shape[-1])
    gt_target = gt_target[..., :min_len]
    pred = pred[..., :min_len].float().squeeze(1)
    baseline = baseline[..., :min_len].float()

    return low_input, baseline, pred, gt_target


def safe_audio_for_streamlit(tensor: torch.Tensor, sample_rate: int):
    tensor = tensor.detach().cpu().float().squeeze().contiguous()
    if tensor.ndim == 0:
        tensor = tensor.unsqueeze(0)
    return make_wav_bytes(tensor, sample_rate)


def main():
    st.title("Any-to-Any Audio Super-Resolution Dashboard")
    st.caption(
        "Compare the low-resolution input, a direct interpolation baseline, and the model reconstruction against the original high-resolution audio."
    )

    with st.sidebar:
        st.header("Model & Task")
        model_dir = Path("models")
        checkpoints = sorted([
            p.name for p in model_dir.glob("*.pt") if p.is_file()
        ]) if model_dir.exists() else []

        if not checkpoints:
            st.error("No checkpoint files were found in the models/ directory.")
            st.stop()

        checkpoint_name = st.selectbox("Checkpoint", options=checkpoints, index=0)
        checkpoint_path = str(model_dir / checkpoint_name)

        task_preset = st.selectbox(
            "Super-Resolution Task",
            options=[
                "8k to 16k",
                "8k to 24k",
                "8k to 48k",
                "12k to 24k",
                "12k to 48k",
                "24k to 48k",
                "Custom",
            ],
            index=0,
        )

        if task_preset == "Custom":
            low_sr = st.number_input("Low SR", min_value=8000, max_value=48000, value=8000, step=1000)
            high_sr = st.number_input("High SR", min_value=8000, max_value=48000, value=16000, step=1000)
            if low_sr >= high_sr:
                st.warning("Custom low_sr must be lower than high_sr.")
        else:
            src_text, dst_text = task_preset.split(" to ")
            low_sr = int(src_text.replace("k", "000"))
            high_sr = int(dst_text.replace("k", "000"))

        st.divider()
        st.header("Audio Input")
        source_mode = st.radio("Audio Source", options=["Sample Audio", "Upload Custom Audio"], horizontal=True)

        if source_mode == "Sample Audio":
            sample_files = scan_sample_audio("Data")
            if not sample_files:
                st.warning("No 48 kHz .wav or .flac files were found under the Data/ folder.")
                sample_selection = None
            else:
                st.caption("Only 48 kHz sample files are shown here to match the repo’s canonical evaluation setup.")
                sample_selection = st.selectbox(
                    "Choose sample",
                    options=[str(Path(path).resolve()) for path in sample_files],
                    index=0,
                )
        else:
            sample_selection = st.file_uploader(
                "Upload audio",
                type=["wav", "flac", "mp3", "m4a", "ogg"],
                accept_multiple_files=False,
            )

    st.subheader("Selected configuration")
    st.write(f"Checkpoint: {checkpoint_name}")
    st.write(f"Input sampling rate: {human_readable_sr(low_sr)}")
    st.write(f"Target sampling rate: {human_readable_sr(high_sr)}")

    try:
        if source_mode == "Sample Audio":
            if sample_selection is None:
                st.info("Select or upload an audio file to begin.")
                st.stop()
            gt_waveform, gt_sr = load_audio_from_path(sample_selection)
        else:
            if sample_selection is None:
                st.info("Select or upload an audio file to begin.")
                st.stop()
            gt_waveform, gt_sr = load_audio_from_uploaded(sample_selection)

        if gt_sr <= 0:
            raise ValueError("Audio file could not be read with a valid sample rate.")
        if gt_sr != CANONICAL_HIGH_SR:
            raise ValueError(
                f"This project requires a 48 kHz ground-truth audio file for evaluation. "
                f"The selected file is {gt_sr} Hz. Please resample it to 48 kHz before processing."
            )

        st.write(f"Native file sample rate: {human_readable_sr(gt_sr)}")
        st.write(f"Canonical evaluation reference rate: {human_readable_sr(CANONICAL_HIGH_SR)}")
        st.write("The app is using only 48 kHz ground-truth files to match the repo’s evaluation setup.")
        st.write(f"Reference waveform shape after canonical resample: {tuple(gt_waveform.shape)}")

        process_clicked = st.button("Process Audio", use_container_width=True)
        if not process_clicked:
            st.info("Click the button to generate the interpolation baseline and model prediction.")
            st.stop()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            model = load_model(checkpoint_path, device)
        except Exception as exc:
            st.error(f"Model load failed: {exc}")
            st.stop()

        low_input, baseline, prediction, gt_waveform = build_audio_outputs(
            gt_waveform=gt_waveform,
            gt_sr=gt_sr,
            low_sr=low_sr,
            high_sr=high_sr,
            model=model,
            device=device,
        )

        st.success("Processing complete. Results are below.")

        model_lsd = compute_lsd_base10(prediction, gt_waveform)
        baseline_lsd = compute_lsd_base10(baseline, gt_waveform)

        visqol_sr = max(high_sr, 16000)
        model_visqol = compute_visqol_mos(
            torchaudio.functional.resample(gt_waveform, high_sr, visqol_sr),
            torchaudio.functional.resample(prediction, high_sr, visqol_sr),
            visqol_sr,
        )
        baseline_visqol = compute_visqol_mos(
            torchaudio.functional.resample(gt_waveform, high_sr, visqol_sr),
            torchaudio.functional.resample(baseline, high_sr, visqol_sr),
            visqol_sr,
        )

        st.markdown("---")
        st.subheader("Quality Metrics")
        metric_cols = st.columns(4)
        metric_cols[0].metric("LSD (Model)", f"{model_lsd:.4f}", delta="lower is better")
        metric_cols[1].metric("LSD (Baseline)", f"{baseline_lsd:.4f}", delta="lower is better")
        metric_cols[2].metric(
            "ViSQOL (Model)",
            "N/A" if model_visqol is None else f"{model_visqol:.2f}",
            delta="higher is better" if model_visqol is not None else None,
        )
        metric_cols[3].metric(
            "ViSQOL (Baseline)",
            "N/A" if baseline_visqol is None else f"{baseline_visqol:.2f}",
            delta="higher is better" if baseline_visqol is not None else None,
        )
        if model_visqol is None and baseline_visqol is None:
            st.caption("ViSQOL is not installed in this environment, so only LSD is reported.")
        st.markdown("---")

        left_col, right_col = st.columns([1.2, 1.2])
        with left_col:
            st.subheader("Input (Low-Resolution)")
            st.audio(safe_audio_for_streamlit(low_input, low_sr), format="audio/wav")
            st.pyplot(plot_spectrogram(low_input, low_sr, "Low-Res Input Spectrogram"))
        with right_col:
            st.subheader("Ground Truth (Original)")
            st.audio(safe_audio_for_streamlit(gt_waveform, high_sr), format="audio/wav")
            st.pyplot(plot_spectrogram(gt_waveform, high_sr, "Ground Truth Spectrogram"))

        st.markdown("---")
        baseline_col, pred_col = st.columns(2)

        with baseline_col:
            st.subheader("Baseline (Torch Resampled)")
            st.audio(safe_audio_for_streamlit(baseline, high_sr), format="audio/wav")
            st.pyplot(plot_spectrogram(baseline, high_sr, "Baseline Spectrogram"))

        with pred_col:
            st.subheader("Prediction (Model Output)")
            st.audio(safe_audio_for_streamlit(prediction, high_sr), format="audio/wav")
            st.pyplot(plot_spectrogram(prediction, high_sr, "Model Prediction Spectrogram"))

    except Exception as exc:
        st.error(f"Processing error: {exc}")


if __name__ == "__main__":
    main()

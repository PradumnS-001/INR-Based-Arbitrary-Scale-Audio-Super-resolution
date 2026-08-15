import torch
from torch.nn import functional as F
from torchaudio.functional import resample
import soundfile as sf
import os
from torchmetrics.audio import SignalNoiseRatio, SignalDistortionRatio

# Adjust imports to match your file structure
from data import tr_loader, calc_opcs
from configs import *
from models import ImprovedLISA
from extraUtils.loss import WaveLoss, log_spectral_distance, compute_audio_ssim
from extraUtils.layers import ModelEMA

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    supported = torch.cuda.is_bf16_supported()
    cast_type = torch.bfloat16 if supported else torch.float16
    torch.backends.cudnn.benchmark = False

    waveloss = WaveLoss(eps=loss_eps, pow_fac=loss_pow_fac).to(device)
    model = ImprovedLISA(opcs=calc_opcs(tr_loader)).to(device)
    print(f"OPCS: {model.opcs.item()}")
    ema = ModelEMA(model=model)
    
    # Lower initial LR to prevent the momentum bounce
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=wdc)
    scaler = torch.amp.GradScaler(device=device, enabled=not supported)
    
    # The Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=30, gamma=0.2)
    
    frozen_batches = []
    
    # 1. Capture Exactly 1 Batch
    with torch.no_grad():
        for i, (lr_wav, hr_wav) in enumerate(tr_loader):
            lr_wav = lr_wav.to(device, non_blocking=True)
            hr_wav = hr_wav.to(device, non_blocking=True)
            
            scale = 6.0 
            hsr_new = int(low_sampling_rate * scale)
            hr_wav = resample(hr_wav, high_sampling_rate, hsr_new)
            lr_wave_base = resample(lr_wav, low_sampling_rate, hsr_new)
            
            frozen_batches.append((lr_wav, hr_wav, lr_wave_base, scale, hsr_new))
            break
                
    print(f"Captured {len(frozen_batches)} batch. CRITICAL: Ensure do_perturbation=False in your model for this test.")

    best_lsd = float('inf')
    best_model_state = None
    
    # 2. Overfit Loop
    for epoch in range(100):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        
        lr_wav, hr_wav, lr_wave_base, scale, hsr_new = frozen_batches[0]
        
        with torch.autocast(device_type=device.split(':')[0], dtype=cast_type):
            # Model returns exactly one tensor now
            pred = model(lr_wav, scale=scale)
            
            min_len = min(pred.shape[-1], hr_wav.shape[-1])
            pred, hr_wav_c = pred[..., :min_len], hr_wav[..., :min_len]
            lr_wave_base_c = lr_wave_base[..., :min_len]
            
            pred = pred + lr_wave_base_c
            
            loss_mssl = mssl_wt * waveloss(pred, hr_wav_c, scale)
            loss_anchor = l1_wt * F.l1_loss(pred, hr_wav_c) 
            
            loss = loss_mssl + loss_anchor
        
        # Backprop
        if supported:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            scaler.step(optimizer)
            scaler.update()
            
        ema.update(model)
        scheduler.step()
        
        # Save based on MSSL to track spectral convergence
        if loss_mssl.item() < best_lsd:
            best_lsd = loss_mssl.item()
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Total: {loss.item():.4f} | MSSL: {loss_mssl.item():.4f} | L1: {loss_anchor.item():.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    # 3. Final Evaluation
    print("\n--- Training Complete. Evaluating Best Memorized State ---")
    model.load_state_dict(best_model_state)
    model.eval()
    ema.apply_shadow(model)
    
    snr_metric = SignalNoiseRatio().to(device)
    sdr_metric = SignalDistortionRatio().to(device)
    
    with torch.no_grad():
        with torch.autocast(device_type=device.split(':')[0], dtype=cast_type):
            pred = model(lr_wav, scale=scale)
            pred = pred[..., :min_len] + lr_wave_base_c
            
            final_lsd = log_spectral_distance(pred, hr_wav_c).item()
            final_ssim = compute_audio_ssim(pred, hr_wav_c, sample_rate=hsr_new)
            
            snr_metric.update(pred.squeeze(1), hr_wav_c.squeeze(1))
            sdr_metric.update(pred.squeeze(1), hr_wav_c.squeeze(1))
            
            final_snr = snr_metric.compute().item()
            final_sdr = sdr_metric.compute().item()
            
            print(f"Final Memorized LSD:  {final_lsd:.4f} (Target: < 0.9)")
            print(f"Final Memorized SSIM: {final_ssim:.4f} (Target: > 0.8)")
            print(f"Final Memorized SNR:  {final_snr:.4f} dB")
            print(f"Final Memorized SDR:  {final_sdr:.4f} dB")
            
            os.makedirs('audio', exist_ok=True)
            sf.write(os.path.join('audio',"target_overfit.wav"), hr_wav_c.cpu()[0,...].mT.numpy(), hsr_new)
            
            save = pred.cpu()[0,...].mT
            maxv = max(save.abs().max().item(), 1)
            save /= maxv
            sf.write(os.path.join('audio',"pred_overfit.wav"), save.contiguous().numpy(), hsr_new)
            print("Saved audio files to disk. Go listen to them!")

if __name__ == "__main__":
    main()
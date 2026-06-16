import torch
from torch.nn import functional as F
from torchaudio.functional import resample
import soundfile
import os

# Adjust imports to match your file structure
from data import tr_loader
from configs import *
from models import ImprovedLISA
from extraUtils.loss import WaveLoss, ModelEMA, log_spectral_distance, compute_audio_ssim

torch.autograd.detect_anomaly(True)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    supported = torch.cuda.is_bf16_supported()
    cast_type = torch.bfloat16 if supported else torch.float16
    torch.backends.cudnn.benchmark = False

    waveloss = WaveLoss().to(device)
    model = ImprovedLISA().to(device)
    ema = ModelEMA(model=model)
    
    # Lower initial LR to prevent the momentum bounce
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=wdc)
    scaler = torch.amp.GradScaler(device=device, enabled=not supported)
    
    # The Scheduler: Drops LR by 50% if the loss stagnates for 15 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=30,gamma=0.2)
    
    frozen_batches = []
    
    # 1. Capture Exactly 1 Batch
    with torch.no_grad():
        for i, (lr_wav, hr_wav) in enumerate(tr_loader):
            lr_wav = lr_wav.to(device, non_blocking=True)
            hr_wav = hr_wav.to(device, non_blocking=True)
            
            scale = 6.0 
            hsr_new = int(low_sampling_rate * scale)
            hr_wav = resample(hr_wav, high_sampling_rate, hsr_new)
            lr_wave_base = resample(lr_wav,low_sampling_rate,hsr_new)
            
            frozen_batches.append((lr_wav, hr_wav, lr_wave_base, scale, hsr_new))
            break
                
    print(f"Captured {len(frozen_batches)} batch. CRITICAL: Ensure do_perturbation=False in your model for this test.")

    best_l1 = float('inf')
    best_model_state = None
    
    # 2. Overfit Loop
    for epoch in range(200):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        
        lr_wav, hr_wav, lr_wave_base, scale, hsr_new = frozen_batches[0]
        
        with torch.autocast(device_type=device.split(':')[0], dtype=cast_type):
            # For a strict overfit test, we only need one forward pass. 
            # If your model outputs two (predA, predB) for repel loss, we just use one to check capacity.
            pred = model(lr_wav, scale=scale)
            
            min_len = min(pred.shape[-1], hr_wav.shape[-1])
            pred, hr_wav_c = pred[..., :min_len], hr_wav[..., :min_len]
            lr_wave_base_c = lr_wave_base[..., :min_len]
            
            pred = pred + lr_wave_base_c
            
            loss_mssl = waveloss(pred, hr_wav_c)
            # Weight L1 heavily to force phase alignment
            loss_anchor = F.l1_loss(pred, hr_wav_c) * 8.0 
            
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
        scheduler.step() # Feed total loss to the plateau scheduler
        
        # Save best model based on L1 Phase Alignment
        if loss_anchor.item() < best_l1:
            best_l1 = loss_anchor.item()
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Total: {loss.item():.4f} | MSSL: {loss_mssl.item():.4f} | L1(x10): {loss_anchor.item():.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    # 3. Final Evaluation on the Memorized Batch
    print("\n--- Training Complete. Evaluating Best Memorized State ---")
    model.load_state_dict(best_model_state)
    model.eval() # Turns off gamma noise and ensures deterministic output
    ema.apply_shadow(model)
    
    with torch.no_grad():
        with torch.autocast(device_type=device.split(':')[0], dtype=cast_type):
            pred = model(lr_wav, scale=scale)
            pred = pred[..., :min_len] + lr_wave_base_c
            
            final_lsd = log_spectral_distance(pred, hr_wav_c).item()
            final_ssim = compute_audio_ssim(pred, hr_wav_c, sample_rate=hsr_new)
            
            print(f"Final Memorized LSD:  {final_lsd:.4f} (Target: < 0.5)")
            print(f"Final Memorized SSIM: {final_ssim:.4f} (Target: > 0.95)")
            
            soundfile.write(os.path.join('audio',"target_overfit.wav"), hr_wav_c.cpu()[0,...].mT, hsr_new)
            soundfile.write(os.path.join('audio',"pred_overfit.wav"), pred.cpu()[0,...].mT, hsr_new)
            print("Saved audio files to disk. Go listen to them!")

if __name__ == "__main__":
    main()
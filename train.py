import os
import gc
import random
import numpy as np
import torch
import torchaudio
import torch.multiprocessing as mp
import torch.nn.functional as F
from tqdm import tqdm
from torchmetrics.audio import SignalNoiseRatio
from torch.utils.tensorboard import SummaryWriter
import auraloss

from configs import *
from data import tr_loader, val_loader
from models import SIRIUS
from components import ModelEMA
from functions import log_spectral_distance, set_requires_grad, generator_hinge_loss, discriminator_hinge_loss
from encodec.msstftd import MultiScaleSTFTDiscriminator

def main():
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    
    torch.backends.cudnn.benchmark = False
    os.makedirs('models', exist_ok=True)
    os.makedirs('runs', exist_ok=True)
    
    writer = SummaryWriter(log_dir='runs/sirius_run_01')
    global_step = 0
    
    ## Values Obtained from the calculate_stats function in data.py
    mean_data = 0
    std_data_val = 0.0594
    
    print(f"Data Mean: {mean_data}, Data Std: {std_data_val}")
    
    model = SIRIUS(mean=mean_data, std=std_data_val).to(device)
    ema = ModelEMA(model, decay=ema_wt)
    
    if do_adversarial:
        disc = MultiScaleSTFTDiscriminator(filters=filters).to(device)

    else:
        disc = None
        
    mssl = auraloss.freq.MultiResolutionSTFTLoss(
                                fft_sizes=[2048, 1024, 512, 256, 128],
                                hop_sizes=[512, 256, 128, 64, 32],
                                win_lengths=[2048, 1024, 512, 256, 128],
                                sample_rate=max_target_sr,
                                perceptual_weighting=True
                            )


    opt_G = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.75, 0.9), weight_decay=wdc)
    if do_adversarial:
        opt_D = torch.optim.AdamW(disc.parameters(), lr=lr*2, betas=(0.5, 0.9), weight_decay=wdc/10)

    # Scheduler Selection
    cons_sch_G = torch.optim.lr_scheduler.ConstantLR(opt_G, factor=1.0, total_iters=10)
    cos_sch_G = torch.optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max= epochs-step_size, eta_min=lr/100)
    scheduler_G = torch.optim.lr_scheduler.SequentialLR(opt_G, schedulers=[cons_sch_G, cos_sch_G], milestones=[step_size])
    
    cons_sch_D = torch.optim.lr_scheduler.ConstantLR(opt_D, factor=1.0, total_iters=10)
    cos_sch_D = torch.optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max= epochs-step_size, eta_min=lr/50)
    scheduler_D = torch.optim.lr_scheduler.SequentialLR(opt_D, schedulers=[cons_sch_D, cos_sch_D], milestones=[step_size])

    snr_metric = SignalNoiseRatio().to(device)
    
    print(f"Training {low_sampling_rate // 1000}kHz --> {max_target_sr // 1000}kHz @ {scale_res} scale-resolution")
    print("Precomputing Resample Filter Banks...")
    resamplers = {}
    # Use integer division for max scale to avoid float drift
    max_scale_int = (scale_res * max_target_sr) // low_sampling_rate

    for s in range(scale_res, max_scale_int + 1):
        # Multiply FIRST, then floor divide. This completely bypasses floating-point precision issues.
        target_freq = (low_sampling_rate * s) // scale_res
        
        # Key the dictionary by the integer 's' instead of the calculated frequency
        resamplers[s] = torchaudio.transforms.Resample(
            orig_freq=high_sampling_rate, new_freq=target_freq
        ).to(device=device, non_blocking=True)
        
    resample_to_max = torchaudio.transforms.Resample(
        orig_freq=high_sampling_rate, new_freq=max_target_sr
    ).to(device=device, non_blocking=True)
    resample_to_min = torchaudio.transforms.Resample(
        orig_freq=high_sampling_rate, new_freq=low_sampling_rate
    ).to(device=device, non_blocking=True)
    print("Training Started")

    for epoch in range(epochs):
        model.train()
        if do_adversarial: disc.train()

        pbar = tqdm(tr_loader, desc=f"Epoch {epoch}")
        is_threshold = random.random() < thershold

        for step, hr_wav in enumerate(pbar):
            hr_wav = hr_wav.to(device)

            # Route 1: Target Resolution (Always max_target_sr)
            hr_target = resample_to_max(hr_wav) if high_sampling_rate != max_target_sr else hr_wav
            
            # Route 2: Input Resolution (Randomly sampled between low_sr and max_target_sr)
            s = np.random.randint(scale_res, max_scale_int + 1)
            in_sr = (low_sampling_rate * s) // scale_res
            if s in resamplers:
                lr_input = resamplers[s](hr_wav)
            else:
                lr_input = torchaudio.functional.resample(hr_wav, high_sampling_rate, in_sr)

            # --- GENERATOR STEP ---
            if do_adversarial: set_requires_grad([disc], False)
            
            hat_x = model(lr_input, in_sr, max_target_sr)
            
            min_len = min(hat_x.shape[-1], hr_target.shape[-1])
            hat_x, hr_target = hat_x[..., :min_len], hr_target[..., :min_len]

            loss_l1 = F.l1_loss(hat_x, hr_target)

            if do_adversarial:
                loss_percp = mssl(hat_x, hr_target)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits_fake, _ = disc(hat_x)
                    loss_hinge_g = generator_hinge_loss(logits_fake)
                loss_G = (l1_weight * loss_l1) + (mssl_weight * loss_percp) + (adv_weight * loss_hinge_g * (epoch > 0))
            else:
                loss_spec = mssl(hat_x, hr_target)
                loss_G = (l1_weight * loss_l1) + (mssl_weight * loss_spec)

            loss_G = loss_G / update_step
            loss_G.backward()

            # --- DISCRIMINATOR STEP ---
            if do_adversarial and is_threshold and epoch > 0:
                set_requires_grad([disc], True)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits_real, _ = disc(hr_target)
                    logits_fake_det, _ = disc(hat_x.detach())
                    loss_D = discriminator_hinge_loss(logits_real, logits_fake_det) / update_step
                loss_D.backward()

            if (step + 1) % update_step == 0 or (step + 1) == len(tr_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                opt_G.step()
                opt_G.zero_grad()
                
                writer.add_scalar('Train/Loss_G', loss_G.item() * update_step, global_step)
                writer.add_scalar('Train/Loss_L1', loss_l1.item(), global_step)
                
                if do_adversarial:
                    if epoch > 0 and is_threshold:
                        torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm)
                        opt_D.step()
                        opt_D.zero_grad()
                        writer.add_scalar('Train/Loss_D', loss_D.item() * update_step, global_step)
                        
                    writer.add_scalar('Train/Loss_Percp', loss_percp.item(), global_step)
                    is_threshold = random.random() < thershold
                else:
                    writer.add_scalar('Train/Loss_MSSL', loss_spec.item(), global_step)
                
                ema.update(model)
                global_step += 1
                
            if step % 10 == 0:
                loss_G = loss_G.detach()
                pbar.set_postfix({"G_loss": loss_G.item() * update_step})
        torch.cuda.empty_cache()

        scheduler_G.step()
        if do_adversarial: scheduler_D.step()

        # --- VALIDATION ---
        model.eval()
        ema.apply_shadow(model)
        snr_metric.reset()
        avg_lsd = 0.0

        with torch.no_grad():
            for hr_wav in val_loader:
                hr_wav = hr_wav.to(device)
                hr_target = resample_to_max(hr_wav) if high_sampling_rate != max_target_sr else hr_wav
                
                # Test exactly at low_sampling_rate mapping to max_target_sr
                lr_input = resample_to_min(hr_wav)
                
                pred = model(lr_input, low_sampling_rate, max_target_sr)
                min_len = min(pred.shape[-1], hr_target.shape[-1])
                pred, hr_target = pred[..., :min_len], hr_target[..., :min_len]

                snr_metric(pred, hr_target)
                avg_lsd += log_spectral_distance(pred, hr_target).item()
                
        ema.restore(model)

        current_lsd = avg_lsd / len(val_loader)
        current_snr = snr_metric.compute().item()
        
        writer.add_scalar('Val/LSD', current_lsd, epoch)
        writer.add_scalar('Val/SNR', current_snr, epoch)
        print(f"Epoch {epoch} | Val LSD: {current_lsd:.4f} | Val SNR: {current_snr:.2f}")
        
        torch.save({
            'model': model.state_dict(),
            'ema': ema.shadow
        }, os.path.join('models', "sirius_last_model.pt"))

        gc.collect()
        torch.cuda.empty_cache()

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
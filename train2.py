import os
import gc
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import random
import numpy as np
from tqdm import tqdm
import itertools
from torchmetrics.audio import SignalNoiseRatio
from torch.utils.tensorboard import SummaryWriter
import torchaudio

from configs import *
from data import tr_loader, val_loader, calc_opcs
from models import Encoder, INRDecoder
from extraUtils.layer import ModelEMA
from extraUtils.loss import (
    MultiScaleSpectralLoss, log_spectral_distance, kullback_liebler_divergence,
    ganin_scheduler
)
from loss_ext import generator_adv_losses, compute_discriminator_hinge, DynamicSparseBalancer
from encodec.msstftd import MultiScaleSTFTDiscriminator

def set_requires_grad(nets, requires_grad=False):
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = False
    os.makedirs('models', exist_ok=True)
    os.makedirs('runs', exist_ok=True)
    
    writer = SummaryWriter(log_dir='runs/lisa_run_01')
    global_step = 0

    encoder = Encoder().to(device, non_blocking=True)
    mean_data, std_data = calc_opcs(tr_loader)
    std_data = std_data.item()
    huber_delta = 0.01 * std_data if std_data > 0 else 1.0
    decoder = INRDecoder(mean=mean_data, std=std_data).to(device, non_blocking=True)
    
    print(mean_data, std_data)
    print(device)
    
    disc_1x = MultiScaleSTFTDiscriminator(filters=filters, 
                                          n_ffts=[1024, 256, 64], 
                                          hop_lengths=[256, 64, 16],
                                          win_lengths=[1024, 256, 64]).to(device, non_blocking=True)
    disc_2x = MultiScaleSTFTDiscriminator(filters=filters, n_ffts=[2048, 512, 128], 
                                          hop_lengths=[512, 128, 32],
                                          win_lengths=[2048, 512, 128]).to(device, non_blocking=True)
    disc_3x = MultiScaleSTFTDiscriminator(filters=filters, n_ffts=[3072, 768, 192], 
                                          hop_lengths=[768, 192, 48],
                                          win_lengths=[3072, 768, 192]).to(device, non_blocking=True)
    
    muon_params = [p for p in decoder.parameters() if p.ndim >= 2]
    adam_params = [p for p in decoder.parameters() if p.ndim < 2]

    opt_G_muon = torch.optim.Muon(muon_params, lr=lr * muon_scalar, momentum=0.95) 
    opt_G_adam = torch.optim.Adam(
        itertools.chain(encoder.parameters(), adam_params), 
        lr=lr, betas=(0.75, 0.99)
    )
    opt_D = torch.optim.Adam(itertools.chain(disc_1x.parameters(), disc_2x.parameters(), disc_3x.parameters()), lr=lr*2, betas=(0.5, 0.9))

    scheduler_G_muon = torch.optim.lr_scheduler.StepLR(opt_G_muon, step_size=step_size, gamma=gamma)
    scheduler_G_adam = torch.optim.lr_scheduler.StepLR(opt_G_adam, step_size=step_size, gamma=gamma)
    scheduler_D = torch.optim.lr_scheduler.StepLR(opt_D, step_size=step_size, gamma=gamma)
    ema = ModelEMA(decoder, decay=0.999)

    mssl = MultiScaleSpectralLoss().to(device, non_blocking=True)
    snr_metric = SignalNoiseRatio().to(device, non_blocking=True)

    balancer = DynamicSparseBalancer(base_weights=base_weights)
    best_lsd = float('inf')
    
    print("Precomputing Resample Filter Banks...")
    max_scale_int = int(scale_res * (max_target_sr / low_sampling_rate))
    resamplers = {}
    for s in range(scale_res, max_scale_int + 1):
        target_freq = int(low_sampling_rate * (s / scale_res))
        resamplers[target_freq] = torchaudio.transforms.Resample(orig_freq=high_sampling_rate, new_freq=target_freq).to(device=device, non_blocking=True)
    print("Training Started")

    for epoch in range(epochs):
        encoder.train()
        decoder.train()
        disc_1x.train()
        disc_2x.train()
        disc_3x.train()

        epoch_loss = 0
        pbar = tqdm(tr_loader, desc=f"Epoch {epoch}")
        ganin_factor = ganin_scheduler(epoch)
        use_adv = random.random() < thershold

        for step, (lr_wav, hr_wav) in enumerate(pbar):
            lr_wav = lr_wav.to(device, non_blocking=True)
            hr_wav = hr_wav.to(device, non_blocking=True)

            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                
                mu, std = encoder(lr_wav)
                B, D, _ = std.shape
                eps = torch.randn(B, D, 1).to(device, non_blocking=True)
                z = mu + eps * std
                beta = min(1, global_step / beta_steps) * kld_wt
                loss_kld = beta*(kullback_liebler_divergence(mu, std).clamp(min=min_kld)) / update_step

            loss_kld.backward(retain_graph=True)

            scale_arb = np.random.randint(scale_res, int(scale_res * max_target_sr / low_sampling_rate + 1)) / scale_res
            hsr_arb = int(low_sampling_rate * scale_arb)
            with torch.no_grad():
                # hr_arb = resample(hr_wav, high_sampling_rate, hsr_arb)
                hr_arb = resamplers[hsr_arb](hr_wav)
            
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                hat_x_arb = decoder(z, scale=scale_arb)
                min_len_arb = min(hat_x_arb.shape[-1], hr_arb.shape[-1])
                hat_x_arb, hr_arb = hat_x_arb[..., :min_len_arb], hr_arb[..., :min_len_arb]

                loss_mssl = mssl(hat_x_arb, hr_arb)
                loss_huber = F.huber_loss(hat_x_arb, hr_arb, delta=huber_delta)

            active_losses = {'mssl': loss_mssl, 'huber': loss_huber}
            active_outputs = {'mssl': hat_x_arb, 'huber': hat_x_arb}

            if use_adv:
                scale_choice = random.choice([1, 2, 3])
                hsr_fixed = int(low_sampling_rate * scale_choice)
                
                with torch.no_grad():
                    # hr_fixed = resample(hr_wav, high_sampling_rate, hsr_fixed)
                    hr_fixed = resamplers[hsr_fixed](hr_wav)

                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    hat_x_fixed = decoder(z, scale=float(scale_choice))
                    min_len_fixed = min(hat_x_fixed.shape[-1], hr_fixed.shape[-1])
                    hat_x_fixed, hr_fixed = hat_x_fixed[..., :min_len_fixed], hr_fixed[..., :min_len_fixed]

                    disc = {1: disc_1x, 2: disc_2x, 3: disc_3x}[scale_choice]
                    tag = f"{scale_choice}x"

                    logits_real, fmaps_real = disc(hr_fixed)
                    logits_fake, fmaps_fake = disc(hat_x_fixed)

                    loss_hinge, loss_fm = generator_adv_losses(logits_fake, fmaps_real, fmaps_fake)
                    loss_hinge = loss_hinge
                    loss_fm = loss_fm

                active_losses[f'hinge_{tag}'] = loss_hinge
                active_losses[f'fm_{tag}'] = loss_fm
                active_outputs[f'hinge_{tag}'] = hat_x_fixed
                active_outputs[f'fm_{tag}'] = hat_x_fixed
                
            set_requires_grad([disc_1x, disc_2x, disc_3x], False)
            balanced_loss = balancer.get_balanced_loss(active_losses, active_outputs, ganin_factor) / update_step
            balanced_loss.backward()

            if use_adv:
                set_requires_grad([disc_1x, disc_2x, disc_3x], True)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    loss_D = compute_discriminator_hinge(logits_real, disc(hat_x_fixed.detach())[0]) / update_step
                loss_D.backward()

            if (step + 1) % update_step == 0 or (step + 1) == len(tr_loader):
                
                torch.nn.utils.clip_grad_norm_(itertools.chain(encoder.parameters(), decoder.parameters()), max_norm)
                opt_G_muon.step()
                opt_G_adam.step()
                
                writer.add_scalar('Train/Total_Balanced_Loss', balanced_loss.item(), global_step)
                writer.add_scalar('Train/KLD', loss_kld.item() * update_step / (beta+1e-7), global_step)
                writer.add_scalar('Train/MSSL_Raw', active_losses['mssl'].item() * update_step, global_step)
                writer.add_scalar('Train/Huber_Raw', active_losses['huber'].item() * update_step, global_step)
                
                if use_adv:
                    opt_D.step()
                    writer.add_scalar(f'Train/Hinge_Raw_{tag}', active_losses[f'hinge_{tag}'].item() * update_step, global_step)
                    writer.add_scalar(f'Train/FM_Raw_{tag}', active_losses[f'fm_{tag}'].item() * update_step, global_step)
                    writer.add_scalar('Train/Discriminator_Loss', loss_D.item() * update_step, global_step)
                opt_G_muon.zero_grad()
                opt_G_adam.zero_grad()
                opt_D.zero_grad()
                ema.update(decoder)
                writer.add_scalar('Params/Ganin_Factor', ganin_factor, global_step)
                global_step += 1

            epoch_loss += balanced_loss.item()
            pbar.set_postfix({"G_loss": balanced_loss.item()})

            del hr_arb, hat_x_arb, active_losses, active_outputs, balanced_loss, loss_kld, mu, std, z
            if use_adv:
                del hr_fixed, hat_x_fixed, logits_real, logits_fake, fmaps_real, fmaps_fake, loss_hinge, loss_fm, loss_D
                
            if (step + 1) % update_step == 0 or (step + 1) == len(tr_loader): use_adv = random.random() < thershold
            torch.cuda.empty_cache()

        scheduler_G_muon.step()
        scheduler_G_adam.step()
        scheduler_D.step()

        encoder.eval()
        decoder.eval()
        
        active_weights = {k: v.clone().detach() for k, v in decoder.state_dict().items()}
        ema.apply_shadow(decoder)
        snr_metric.reset()
        avg_lsd = 0.0

        with torch.no_grad():
            
            hsr_val = int(low_sampling_rate * val_scale)
            resamp = resamplers[hsr_val]
            
            for lr_wav, hr_wav in val_loader:
                
                lr_wav, hr_wav = lr_wav.to(device, non_blocking=True), hr_wav.to(device, non_blocking=True)
                hr_wav = resamp(hr_wav)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    mu, std = encoder(lr_wav)
                    pred = decoder(mu, scale=val_scale)
                    
                    min_len = min(pred.shape[-1], hr_wav.shape[-1])
                    pred, hr_wav = pred[..., :min_len], hr_wav[..., :min_len]

                snr_metric(pred, hr_wav)
                avg_lsd += log_spectral_distance(pred, hr_wav).item()
                
        ema.shadow = {k: v.clone().detach() for k, v in decoder.state_dict().items() if v.dtype.is_floating_point}
        decoder.load_state_dict(active_weights)
        del active_weights

        current_lsd = avg_lsd / len(val_loader)
        current_snr = snr_metric.compute().item()
        writer.add_scalar('Val/LSD', current_lsd, epoch)
        writer.add_scalar('Val/SNR', current_snr, epoch)
        print(f"Epoch {epoch} | LSD: {current_lsd:.4f} | SNR: {current_snr:.2f}")

        if current_lsd <= best_lsd:
            best_lsd = current_lsd
            torch.save({
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'ema': ema.shadow,
                'lsd': best_lsd
            }, os.path.join('models', "lisa_best_model_lsd.pt"))
        
        if epoch == epochs - 1:
            torch.save({
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'ema': ema.shadow
            }, os.path.join('models', "lisa_last_model.pt"))

        gc.collect()
        torch.cuda.empty_cache()

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
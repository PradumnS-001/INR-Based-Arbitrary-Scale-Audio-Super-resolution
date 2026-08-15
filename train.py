import torch
from torch.nn import functional as F
from torchaudio.functional import resample
from tqdm import tqdm
import numpy as np
import gc
import os
import soundfile as sf
from torchmetrics.audio import SignalNoiseRatio, SignalDistortionRatio

from data import tr_loader, val_loader, calc_opcs
from configs import *
from models import ImprovedLISA
from extraUtils.loss import WaveLoss, log_spectral_distance, compute_audio_ssim
from extraUtils.misc import count_params

def calc_loss(
    pred:torch.Tensor,
    base:torch.Tensor, 
    scale:float,
    wls:WaveLoss)->torch.Tensor:
    
    mssl = mssl_wt * wls(pred,base,scale)
    l1_anchor = l1_wt * F.l1_loss(pred,base)
    
    return mssl + l1_anchor

def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    supported = torch.cuda.is_bf16_supported()
    print(supported)
    print("Val Scale: ", val_scale)
    cast_type = torch.bfloat16 if supported else torch.float16
    torch.backends.cudnn.benchmark = False

    model = ImprovedLISA(opcs=calc_opcs(tr_loader)).to(device)
    wls = WaveLoss(eps=loss_eps,pow_fac=loss_pow_fac).to(device)
    print(model.opcs.item())
    ours = count_params(model=model)
    print('Our param count / Lisa\'s param count: ',ours / 89000)
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or 'bias' in name:
            no_decay_params.append(param)
        elif 'weight_g' in name:
            no_decay_params.append(param)
        elif 'weight_v' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    optimizer = torch.optim.Adam([
        {'params': decay_params, 'weight_decay': wdc},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=lr,eps=1e-12)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    scalar = torch.amp.GradScaler(device=device, enabled=not supported)
    best_lsd = float('inf')

    for epoch in range(epochs):
        
        model.train()
        pbar = tqdm(tr_loader, desc=f"Epoch {epoch}")
        optimizer.zero_grad(set_to_none=True)
        
        for i, (lr_wav, hr_wav) in enumerate(pbar):
            
            lr_wav:torch.Tensor = lr_wav.to(device, non_blocking=True)
            scale = np.random.randint(scale_res, int(scale_res*high_sampling_rate/low_sampling_rate + 1)) / scale_res
            hsr_new = int(low_sampling_rate * scale)
            with torch.no_grad(): 
                hr_wav = resample(hr_wav, high_sampling_rate, hsr_new)
                lr_wave_base = resample(lr_wav,low_sampling_rate,hsr_new)
            hr_wav:torch.Tensor = hr_wav.to(device, non_blocking=True)
            with torch.autocast(device_type=device, dtype=cast_type):
                pred = model(lr_wav, scale=scale)
                
                min_len = min(pred.shape[-1],hr_wav.shape[-1])
                pred, hr_wav, lr_wave_base = pred[...,:min_len], hr_wav[...,:min_len], lr_wave_base[...,:min_len]
                
            pred = pred.float()
            lr_wave_base, hr_wav = lr_wave_base.float(), hr_wav.float()
            pred = pred + lr_wave_base
            
            loss = calc_loss(pred=pred,base=hr_wav,scale=scale,wls=wls)
            loss = loss / update_step
            
            if supported : loss.backward()
            else : scalar.scale(loss).backward()
            
            pbar.set_postfix({"loss": loss.item() * update_step})
            if (i + 1) % update_step == 0 or (i + 1) == len(tr_loader):
                if not supported: scalar.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                if supported: optimizer.step()
                else:
                    scalar.step(optimizer)
                    scalar.update()
                optimizer.zero_grad(set_to_none=True)
            
        gc.collect()
        torch.cuda.empty_cache()
        
        model.eval()
        snr_metric = SignalNoiseRatio().to(device)
        sdr_metric = SignalDistortionRatio().to(device)
        avg_lsd = 0
        avg_ssim = 0
        avg_base_lsd = 0
        avg_base_ssim = 0
        with torch.no_grad():
            
            for lr_wav, hr_wav in val_loader:
                lr_wav, hr_wav = lr_wav.to(device), hr_wav.to(device)
                
                scale = val_scale
                hsr_new = int(low_sampling_rate * scale)
                hr_wav = resample(hr_wav, high_sampling_rate, hsr_new)
                lr_wave_base = resample(lr_wav,low_sampling_rate,hsr_new)
                
                pred = model(lr_wav, scale=scale)
                min_len = min(pred.shape[-1],hr_wav.shape[-1])
                pred, hr_wav = pred[...,:min_len], hr_wav[...,:min_len]
                lr_wave_base = lr_wave_base[...,:min_len]
                pred += lr_wave_base
                
                avg_lsd += log_spectral_distance(pred, hr_wav).item()
                avg_ssim += compute_audio_ssim(waveform_pred=pred, waveform_target=hr_wav,sample_rate=hsr_new)
                avg_base_lsd += log_spectral_distance(lr_wave_base, hr_wav).item()
                avg_base_ssim += compute_audio_ssim(waveform_pred=lr_wave_base, waveform_target=hr_wav, sample_rate=hsr_new)
                snr_metric.update(pred.squeeze(1), hr_wav.squeeze(1))
                sdr_metric.update(pred.squeeze(1), hr_wav.squeeze(1))
                
        current_val_lsd = avg_lsd / len(val_loader)
        current_val_ssim = avg_ssim / len(val_loader)
        current_base_lsd = avg_base_lsd / len(val_loader)
        current_base_ssim = avg_base_ssim / len(val_loader)
        current_val_snr = snr_metric.compute().item()
        current_val_sdr = sdr_metric.compute().item()
        snr_metric.reset()
        sdr_metric.reset()
        
        print(f"Epoch {epoch} | Val LSD: {current_val_lsd:.4f} (Base: {current_base_lsd:.4f}) | Val SSIM: {current_val_ssim:.4f} (Base: {current_base_ssim:.4f}) | SNR: {current_val_snr:.4f} | SDR: {current_val_sdr:.4f}")
        
        os.makedirs('models', exist_ok=True)
        os.makedirs('audio', exist_ok=True)
        if current_val_lsd <= best_lsd:
            best_lsd = current_val_lsd
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lsd': best_lsd,
                'ssim': current_val_ssim
            }, os.path.join('models',"lisa_best_model_lsd.pth"))
            sf.write(os.path.join('audio',"lisa_best_clip_actual_lsd.wav"), hr_wav.cpu()[0,...].mT.contiguous().numpy(), hsr_new)
            save = pred.cpu()[0,...].mT
            maxv = max(save.abs().max().item(),1)
            save /= maxv
            sf.write(os.path.join('audio',"lisa_best_clip_predicted_lsd.wav"), save.contiguous().numpy(), hsr_new)
            print(f"--> Best model saved with LSD: {best_lsd:.4f}")
            
        if epoch >= scheduler_start: scheduler.step()
        
if __name__ == "__main__":
    main()
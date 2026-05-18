def main():

    import torch
    from torch.nn import functional as F
    from torchaudio.functional import resample
    from torch.utils.data import random_split, DataLoader
    from tqdm import tqdm
    import numpy as np
    import gc

    from data import dataset2x
    from models import LISA
    from extraUtils.loss import MultiScaleSpectralLoss, log_spectral_distance
    from torchmetrics.audio import SignalNoiseRatio

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    torch.backends.cudnn.benchmark = False

    lr = 1e-3
    batch_size = 32
    epochs = 100
    best_lsd = float('inf')

    mssl = MultiScaleSpectralLoss()
    snr_metric = SignalNoiseRatio().to(device)

    tr_len = int(0.9 * len(dataset2x))
    val_len = len(dataset2x) - tr_len
    tr_set, val_set = random_split(dataset2x, [tr_len, val_len])

    tr_loader = DataLoader(tr_set, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True, prefetch_factor=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=4, persistent_workers=True, pin_memory=True, prefetch_factor=4)

    model = LISA().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    best_lsd = float('inf')

    for epoch in range(epochs):
        
        model.train()
        epoch_loss = 0
        pbar = tqdm(tr_loader, desc=f"Epoch {epoch}")
        
        for lr_wav, hr_wav in pbar:
            
            lr_wav:torch.Tensor = lr_wav.to(device)
            scale = np.random.randint(25,76) / 25
            hsr_new = int(8000 * scale)
            with torch.no_grad(): hr_wav = resample(hr_wav, 24000, hsr_new)
            
            hr_wav:torch.Tensor = hr_wav.to(device)
            optimizer.zero_grad()
            
            pred = model(lr_wav, scale=scale)
            min_len = min(pred.shape[-1],hr_wav.shape[-1])
            pred, hr_wav = pred[...,:min_len], hr_wav[...,:min_len]
            
            loss:torch.Tensor = 1 * mssl(pred, hr_wav) + 75 * F.l1_loss(pred, hr_wav)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.001)
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            
        gc.collect()
        torch.cuda.empty_cache()
        
        avg_train_loss = epoch_loss / len(tr_loader)
        
        model.eval()
        snr_metric.reset()
        avg_lsd = 0
        with torch.no_grad():
            for lr_wav, hr_wav in val_loader:
                lr_wav, hr_wav = lr_wav.to(device), hr_wav.to(device)
                
                scale = 3.0
                hsr_new = int(8000 * scale)
                hr_wav = resample(hr_wav, 24000, hsr_new)
                
                pred = model(lr_wav, scale=scale)
                min_len = min(pred.shape[-1],hr_wav.shape[-1])
                pred, hr_wav = pred[...,:min_len], hr_wav[...,:min_len]
                
                snr_metric(pred, hr_wav)
                avg_lsd += log_spectral_distance(pred, hr_wav).item()
                
        current_val_snr = snr_metric.compute().item()
        current_val_lsd = avg_lsd / len(val_loader)
        
        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val SNR: {current_val_snr:.2f} | Val LSD: {current_val_lsd:.4f}")
        
        if current_val_lsd < best_lsd:
            best_lsd = current_val_lsd
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lsd': best_lsd,
                'snr': current_val_snr
            }, "lisa_best_model.pt")
            print(f"--> Best model saved with LSD: {best_lsd:.4f}")
        scheduler.step()
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"lisa_checkpoint_epoch_{epoch}.pt")

    with torch.no_grad():
        model.eval()
        snr_metric.reset()
        avg_lsd = 0
        
        for lr_wav, hr_wav in val_loader:
            lr_wav = lr_wav.to(device)
            scale = 3.0
            hsr_new = int(8000 * scale)
            hr_wav = resample(hr_wav, 24000, hsr_new)
            hr_wav = hr_wav.to(device)
            pred = model(lr_wav, scale=scale)
            snr_metric(pred, hr_wav)
            avg_lsd += log_spectral_distance(pred, hr_wav).item()
            
        print(f"\nFinal Test Results:")
        print(f"Test SNR: {snr_metric.compute().item()/len(val_loader):.2f}")
        print(f"Test LSD: {avg_lsd/len(val_loader):.4f}")
        
if __name__ == "__main__":
    main()
import torch
from torch.nn import functional as F
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm

from data import dataset2x, dataset4x
from models import LISA
from extraUtils.loss import MultiScaleSpectralLoss, log_spectral_distance
from torchmetrics.audio import SignalNoiseRatio

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
torch.backends.cudnn.benchmark = True

lr = 1e-4
batch_size = 16
epochs = 1000
best_lsd = float('inf')

mssl = MultiScaleSpectralLoss()
snr_metric = SignalNoiseRatio().to(device)

tr_len = int(0.9 * len(dataset2x))
val_len = len(dataset2x) - tr_len
tr_set, val_set = random_split(dataset2x, [tr_len, val_len])

tr_loader = DataLoader(tr_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size)

model = LISA().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    pbar = tqdm(tr_loader, desc=f"Epoch {epoch}")
    
    for lr_wav, hr_wav in pbar:
        lr_wav, hr_wav = lr_wav.to(device), hr_wav.to(device)
        optimizer.zero_grad()
        
        pred = model(lr_wav, scale=hr_wav.shape[-1]//lr_wav.shape[-1])
        loss = 0.5 * mssl(pred, hr_wav) + F.l1_loss(pred, hr_wav)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})
    
    avg_train_loss = epoch_loss / len(tr_loader)
    
    model.eval()
    avg_snr = 0
    avg_lsd = 0
    with torch.no_grad():
        for lr_wav, hr_wav in val_loader:
            lr_wav, hr_wav = lr_wav.to(device), hr_wav.to(device)
            pred = model(lr_wav, scale=hr_wav.shape[-1]//lr_wav.shape[-1])
            
            avg_snr += snr_metric(pred, hr_wav).item()
            avg_lsd += log_spectral_distance(pred, hr_wav).item()
            
    current_val_snr = avg_snr / len(val_loader)
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

    if epoch % 10 == 0:
        torch.save(model.state_dict(), f"lisa_checkpoint_epoch_{epoch}.pt")

with torch.no_grad():
    model.eval()
    avg_snr = 0
    avg_lsd = 0
    
    for lr_wav, hr_wav in val_loader:
        lr_wav, hr_wav = lr_wav.to(device), hr_wav.to(device)
        pred = model(lr_wav, scale=hr_wav.shape[-1]//lr_wav.shape[-1])
        
        avg_snr += snr_metric(pred, hr_wav).item()
        avg_lsd += log_spectral_distance(pred, hr_wav).item()
        
    print(f"\nFinal Test Results:")
    print(f"Test SNR: {avg_snr/len(val_loader):.2f}")
    print(f"Test LSD: {avg_lsd/len(val_loader):.4f}")
import torch
import torch.nn.functional as F

## Log Spectral Distance
def log_spectral_distance(y_hat:torch.Tensor, y:torch.Tensor, n_fft = 512)->torch.Tensor:
    """
    Measures the log spectral distance
    """
    window = torch.hann_window(n_fft, device=y.device)
        
    s_hat = torch.stft(y_hat.squeeze(1) if y_hat.ndim > 1 else y_hat, 
                        n_fft, return_complex=True, window=window).abs().pow(2)
    s = torch.stft(y.squeeze(1) if y.ndim > 1 else y, 
                    n_fft, return_complex=True, window=window).abs().pow(2)
    
    log10_s_hat = torch.log10(s_hat + 1e-10)
    log10_s = torch.log10(s + 1e-10)
    
    dist_per_frame = torch.sqrt(torch.mean((log10_s - log10_s_hat)**2, dim=-2))
    return torch.mean(dist_per_frame)

# Helper function to toggle the grads for the discriminator
def set_requires_grad(nets, requires_grad=False):
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
                
def generator_hinge_loss(logits_fake: list[torch.Tensor]) -> torch.Tensor:
    """
    Computes the stable Hinge loss for the generator.
    Expects a list of logits if using a Multi-Scale Discriminator.
    """
    loss_g = 0.0
    for l_fake in logits_fake:
        loss_g += -torch.mean(l_fake)
    
    return loss_g / len(logits_fake)

def discriminator_hinge_loss(
    logits_real: list[torch.Tensor], 
    logits_fake_detached: list[torch.Tensor]
) -> torch.Tensor:
    """
    Computes the stable Hinge loss for the discriminator.
    NOTE: logits_fake_detached MUST be computed from pred.detach() 
    to prevent gradients from flowing back into the generator.
    """
    loss_d = 0.0
    for l_real, l_fake in zip(logits_real, logits_fake_detached):
        # max(0, 1 - D(real)) + max(0, 1 + D(fake))
        loss_d += torch.mean(F.relu(1.0 - l_real)) + torch.mean(F.relu(1.0 + l_fake))
                  
    return loss_d / len(logits_real)
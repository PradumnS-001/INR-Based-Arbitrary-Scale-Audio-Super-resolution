import os
import torch
from torch import nn

## Activation Function Builder
class ActivationFunction(nn.Module):
    
    def __init__(self, act:str | None='relu'):
        super().__init__()
        self.activationFunction = lambda x : x
        act = act.lower()
        
        if act=='gelu': self.activationFunction = nn.GELU(approximate='tanh')
        elif act=='lrelu': self.activationFunction = nn.LeakyReLU(negative_slope=0.2)
        elif act=='elu': self.activationFunction = nn.ELU()
        elif act=='prelu': self.activationFunction = nn.PReLU()
        elif act=='sigmoid': self.activationFunction = nn.Sigmoid()
        elif act=='tanh': self.activationFunction = nn.Tanh()
        elif act=='relu': self.activationFunction = nn.ReLU()
        elif act=='silu': self.activationFunction = nn.SiLU()
        elif act=='softplus': self.activationFunction = nn.Softplus()
        
    def forward(self, x)->torch.Tensor:
        return self.activationFunction(x)
    
## Exponential Moving Average
class ModelEMA:
    def __init__(self, model, decay=0.99):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self.backup = {}

    @torch.no_grad()
    def update(self, model):
        state_dict = model.state_dict()
        for name in self.shadow:
            if self.shadow[name].dtype.is_floating_point:
                self.shadow[name].copy_(
                    self.decay * self.shadow[name] + (1.0 - self.decay) * state_dict[name]
                )
            else:
                self.shadow[name].copy_(state_dict[name])

    @torch.no_grad()
    def apply_shadow(self, model):
        """Backs up online weights and loads EMA weights for validation."""
        self.backup = {k: v.clone().detach() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)

    @torch.no_grad()
    def restore(self, model):
        """Restores the original online weights to resume training."""
        if not self.backup:
            raise RuntimeError("Cannot restore weights without calling apply_shadow first.")
        model.load_state_dict(self.backup, strict=True)
        self.backup = {}
        
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

## To fetch the files
def get_leaf_files(
    path:str, 
    ender:str|tuple[str]='', 
    starter:str|tuple[str]='', 
    container:str='')->tuple[int,list[str]]:
    """
    Returns all the leaf-files in a folder
    """
    path = rf'{path}'
    files = []
    with os.scandir(path=path) as entries:
        
        for entry in entries:
            
            name = os.path.join(path, entry.name)
            if entry.is_file():
                
                files += [name] if (name.startswith(starter) and name.endswith(ender) and container in name) else []
                
            elif entry.is_dir():
                
                files += get_leaf_files(
                    path= name,
                    starter=starter,
                    ender=ender,
                    container=container)[1]
                
            else: files += []
            
    return len(files), files

## Param Count
def count_params(model:nn.Module):
    """Prints a brief summary of the trainable parameter count per module."""
    print("\n" + "="*45)
    print("Model Parameter Summary")
    print("="*45)
    
    total_params = 0
    module_counts = {}
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Group by the top-level module name (e.g., 'macro_encoder', 'alpha_branch')
            module_name = name.split('.')[0]
            module_counts[module_name] = module_counts.get(module_name, 0) + param.numel()
            total_params += param.numel()
            
    for mod, count in module_counts.items():
        print(f"{mod:<25}: {count:,}")
        
    print("-" * 45)
    print(f"{'Total Trainable Params':<25}: {total_params:,}")
    print("="*45 + "\n")
        
    return total_params
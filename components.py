import torch
from torch import nn
import math
import torch.nn.functional as F

class Sine(nn.Module):
    def __init__(self, w0=30.0):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

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
        elif act=='sine': self.activationFunction = Sine()
        else: self.activationFunction = nn.ReLU()
        
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
        
### Model Components

class KernelNet(nn.Module):
    """3-Layer SIREN that parameterizes the continuous convolutional kernel."""
    def __init__(self, in_channels, out_channels, w0=30.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            Sine(w0),
            nn.Linear(32, 32),
            Sine(w0),
            nn.Linear(32, in_channels * out_channels)
        )
        
        # Official SIREN Initialization
        with torch.no_grad():
            self.net[0].weight.uniform_(-1.0, 1.0)
            self.net[0].bias.uniform_(-1.0, 1.0)
            self.net[2].weight.uniform_(-math.sqrt(6/32)/w0, math.sqrt(6/32)/w0)
            self.net[2].bias.uniform_(-1.0, 1.0)
            self.net[4].weight.uniform_(-math.sqrt(6/32)/w0, math.sqrt(6/32)/w0)
            self.net[4].bias.uniform_(-1.0, 1.0)

    def forward(self, rel_pos):
        return self.net(rel_pos)
    
class CKConv1d(nn.Module):
    """
    A sampling-rate invariant convolution layer. Generates discrete weights 
    on-the-fly based on a physical temporal window.
    """
    def __init__(self, in_channels, out_channels, time_window_ms=5.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_window_ms = time_window_ms
        
        self.kernel_net = KernelNet(in_channels, out_channels, w0=30.0)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x, sr):
        # Calculate kernel size based on physical time and current SR
        if self.time_window_ms <= 0.0:
            kernel_size = 1
        else:
            kernel_size = int((self.time_window_ms / 1000.0) * sr)
            if kernel_size % 2 == 0: 
                kernel_size += 1 # Force odd kernel for symmetric padding
            kernel_size = max(1, kernel_size) # Guard rail: never go below 1
            
        # Generate normalized relative positions [-1, 1]
        rel_pos = torch.linspace(-1.0, 1.0, kernel_size, device=x.device).unsqueeze(-1)
        
        # Generate weights: [kernel_size, in_channels * out_channels]
        weights = self.kernel_net(rel_pos) / math.sqrt(kernel_size * self.in_channels)
        
        # Reshape to standard conv1d format: [out_channels, in_channels, kernel_size]
        weights = weights.view(kernel_size, self.out_channels, self.in_channels).permute(1, 2, 0)
        
        padding = kernel_size // 2
        return F.conv1d(x, weights, bias=self.bias, padding=padding)
    
class FiLMSirenLayer(nn.Module):
    """A single SIREN layer modulated by FiLM parameters."""
    def __init__(self, in_features, out_features, w0=30.0, is_first=False):
        super().__init__()
        self.w0 = w0
        self.linear = nn.Linear(in_features, out_features)
        
        # Official SIREN Initialization
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1.0 / in_features, 1.0 / in_features)
            else:
                bound = math.sqrt(6 / in_features) / w0
                self.linear.weight.uniform_(-bound, bound)
                
    def forward(self, x, gamma, beta):
        # Linear projection
        out = self.linear(x)
        # Periodic modulation: sin((gamma + w0) * (Wx + b) + beta)
        return torch.sin((gamma + self.w0) * out + beta)
    
class FiLMPCINR(nn.Module):
    """
    Conditional Implicit Neural Representation with Periodic nonlinearities.
    Applies the same global FiLM modulation to every layer.
    """
    def __init__(self, num_layers=4, hidden_dim=128, w0=30.0):
        super().__init__()
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList()
        # First layer takes 1D coordinate (t_rel)
        self.layers.append(FiLMSirenLayer(1, hidden_dim, w0=w0, is_first=True))
        
        for _ in range(num_layers - 1):
            self.layers.append(FiLMSirenLayer(hidden_dim, hidden_dim, w0=w0, is_first=False))
            
        self.final_linear = nn.Linear(hidden_dim, 1)
        
        with torch.no_grad():
            bound = math.sqrt(6 / hidden_dim) / w0
            self.final_linear.weight.uniform_(-bound, bound)

    def forward(self, t_rel, gammas, betas):
        x = t_rel
        for layer in self.layers:
            # We apply the exact same global gamma and beta to every layer
            x = layer(x, gammas, betas)
            
        return self.final_linear(x)
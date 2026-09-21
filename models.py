import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from configs import encoder_dim, mdim, actfe, ckconv_window

# ==========================================
# 1. Activation Functions & Utilities
# ==========================================

class Sine(nn.Module):
    def __init__(self, w0=30.0):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

def getActivation(act: str = 'relu'):
    act = act.lower()
    if act == 'gelu': return nn.GELU(approximate='tanh')
    elif act == 'lrelu': return nn.LeakyReLU(negative_slope=0.2)
    elif act == 'elu': return nn.ELU()
    elif act == 'silu': return nn.SiLU()
    elif act == 'sine': return Sine()
    return nn.ReLU()

# ==========================================
# 2. Continuous Kernel Convolution (CKConv)
# ==========================================

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
    
class LatentMappingNetwork(nn.Module):
    """
    Maps the local latent triplet + relative time to a SINGLE global FiLM 
    scaling (\gamma) and shifting (\beta) parameter pair applied to all layers.
    Mathematically mirrors the official PiGANMappingNetwork.
    """
    def __init__(self, in_features, hidden_dim=mdim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # The official piGAN output dimension is strictly hidden_dim * 2
        # Half for the global gamma, half for the global beta
        out_features = hidden_dim * 2
        
        layers = []
        layers.extend([nn.Linear(in_features, hidden_dim), nn.LeakyReLU(0.2, inplace=True)])
        nn.init.kaiming_normal_(layers[-2].weight, mode='fan_in', nonlinearity='leaky_relu', a=0.2)
        
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2, inplace=True)])
        nn.init.kaiming_normal_(layers[-2].weight, mode='fan_in', nonlinearity='leaky_relu', a=0.2)
        
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2, inplace=True)])
        nn.init.kaiming_normal_(layers[-2].weight, mode='fan_in', nonlinearity='leaky_relu', a=0.2)
        
        layers.extend([nn.Linear(hidden_dim, out_features)])
        nn.init.kaiming_normal_(layers[-1].weight, mode='fan_in', nonlinearity='leaky_relu', a=0.2)
        
        self.net = nn.Sequential(*layers)
        
        # Official PiGAN stabilization strategy
        with torch.no_grad():
            self.net[-1].weight *= 0.25

    def forward(self, x):
        out = self.net(x)
        
        # Split the last dimension exactly in half
        gammas, betas = out[..., :self.hidden_dim], out[..., self.hidden_dim:]
        gammas = 30.0 * gammas / (1.0 + torch.abs(gammas))
        betas = math.pi * betas / (1.0 + torch.abs(betas))
        
        # Output shape: [B, L_hr, hidden_dim]
        return gammas, betas

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

# ==========================================
# 3. Dynamic Encoder
# ==========================================

class DynamicEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = CKConv1d(1, 16, time_window_ms=7/13*ckconv_window)
        self.act1 = getActivation(actfe)
        
        self.conv2 = CKConv1d(16, 32, time_window_ms=3/13*ckconv_window)
        self.act2 = getActivation(actfe)
        
        self.conv3 = CKConv1d(32, 64, time_window_ms=3/13*ckconv_window)
        self.act3 = getActivation(actfe)
        
        self.conv_out = CKConv1d(64, encoder_dim, time_window_ms=0.0)

    def forward(self, x, sr):
        x = self.act1(self.conv1(x, sr))
        x = self.act2(self.conv2(x, sr))
        x = self.act3(self.conv3(x, sr))
        z = self.conv_out(x, sr)
        return z

# ==========================================
# 5. FiLM PCINR (Decoder)
# ==========================================

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
    def __init__(self, num_layers=4, hidden_dim=mdim, w0=30.0):
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

# ==========================================
# 6. Top-Level Model Wrapper
# ==========================================

class Model(nn.Module):
    def __init__(self, mean=0.0, std=0.0594):
        super().__init__()
        self.register_buffer('data_mean', torch.tensor(mean))
        self.register_buffer('data_std', torch.tensor(std))
        
        self.encoder = DynamicEncoder()
        
        # In Features: Triplet (3 * encoder_dim) + t_rel (1) (Fourier PE dropped)
        in_features = (3 * encoder_dim)
        self.num_inr_layers = 4
        
        self.mapping_net = LatentMappingNetwork(in_features, hidden_dim=mdim)
        self.pcinr = FiLMPCINR(num_layers=self.num_inr_layers, hidden_dim=mdim)

    def forward(self, x, low_sr, high_sr):
        # Derive scale dynamically from the provided sampling rates
        scale = high_sr / low_sr
        
        # 1. Native Input Normalization
        x_norm = (x - self.data_mean) / (self.data_std + 1e-8)
        
        B, _, L_lr = x_norm.shape
        
        # 2. Dynamic Encoder extracts exactly 1 latent per input sample
        z = self.encoder(x_norm, low_sr)
        z_pad = F.pad(z, (1, 1), mode='replicate')
        z_prev = z_pad[:, :, :-2]
        z_curr = z_pad[:, :, 1:-1]
        z_next = z_pad[:, :, 2:]
        
        z_triplet = torch.cat([z_prev, z_curr, z_next], dim=1).transpose(1, 2)
        gammas, betas = self.mapping_net(z_triplet)
        
        # 3. Calculate target grid size
        L_hr = int(L_lr * scale)
        
        # Target timestamps mapped to the LR grid coordinates
        t_mapped = torch.arange(L_hr, device=z.device).float() / scale
        t_mapped = t_mapped.unsqueeze(0).repeat(B, 1)
        
        # Jitter training targets for robust interpolation
        # if self.training:
        #     eta = torch.randn_like(t_mapped) * (0.5 / scale)
        #     t_select = t_mapped + eta
        # else:
        t_select = t_mapped
            
        # 4. Nearest Latent Indices & Relative Time
        idx_i = torch.floor(t_select + 0.5).long().clamp(0, L_lr - 1)
        
        # t_rel is strictly the symmetric fractional offset in [-0.5, 0.5]
        t_rel = (t_mapped - idx_i.float()).unsqueeze(-1) 
        
        # 5. Latent Triplet Extraction (with replicate padding for boundaries)
        #z_pad = F.pad(z, (1, 1), mode='replicate')
        # idx_prev = idx_i
        # idx_curr = idx_i + 1
        # idx_next = idx_i + 2
        
        # def collect(idx):
        #     expanded_idx = idx.unsqueeze(1).expand(-1, z.shape[1], -1)
        #     return torch.gather(z_pad, 2, expanded_idx)

        # Result: [B, L_hr, 3 * encoder_dim]
        expanded_idx = idx_i.unsqueeze(-1).expand(-1, -1, gammas.size(2))
        gammas = torch.gather(gammas, 1, expanded_idx)
        betas = torch.gather(betas, 1, expanded_idx)
        
        # 7. Generate Waveform via PCINR
        out = self.pcinr(t_rel, gammas, betas)
        out = out.transpose(1, 2).contiguous()
        
        # 8. Native Output Denormalization 
        out = (out * self.data_std) + self.data_mean
            
        return out
        
    def load_checkpoint(self, path, device):
        """Standardized checkpoint loader for evaluation wrappers."""
        checkpoint = torch.load(path, map_location=device)
        
        # Prioritize EMA weights if they exist, fallback to standard model weights
        state_key = 'ema' if 'ema' in checkpoint else 'model'
        
        self.load_state_dict(checkpoint[state_key], strict=True)
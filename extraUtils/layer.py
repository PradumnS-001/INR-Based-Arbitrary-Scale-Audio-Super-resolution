import torch
from torch import nn
import torch.nn.functional as F

class Snake(nn.Module):
    
    """
    Snake activation class
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.alpha = nn.Parameter(torch.ones(1))
        
    def forward(self,x):
        
        alpha = torch.where(self.alpha.abs() < 1e-6, 
                                1e-6 * self.alpha.sgn(), 
                                self.alpha)
        alpha = torch.where(alpha == 0, 1e-7, alpha)
        
        exact_res = x + (1 - torch.cos(2 * alpha * x)) / (2 * alpha)
        taylor_res = x + self.alpha * (x**2)
        
        return torch.where(self.alpha.abs() < 1e-6, taylor_res, exact_res)
    
class getActivation(nn.Module):
    
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
        elif act=='snake': self.activationFunction = Snake()
        
    def forward(self, x)->torch.Tensor:
        return self.activationFunction(x)
    
class WeightNormLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, eps=1e-6):
        super().__init__(in_features, out_features, bias)
        self.eps = eps
        self.weight_g = nn.Parameter(torch.ones(out_features, 1))
        nn.init.kaiming_normal_(self.weight)

    def forward(self, x):
        norm = self.weight.norm(dim=1, keepdim=True)
        safe_weight = self.weight * (self.weight_g / (norm + self.eps))
        return F.linear(x, safe_weight, self.bias)
    
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
        
class Swiglu(nn.Module):
    """
    Input: (...,Din)
    Output: (...,Dout)
    """
    def __init__(self, in_features, out_features,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.proj = nn.Linear(in_features=in_features,out_features=out_features*2)
        
    def forward(self, x):
        x = self.proj(x)
        gate, info = x.chunk(2, dim=-1)
        return info * F.silu(gate)
    
class DenseLayer1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer1D, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


class RDB1D(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB1D, self).__init__()
        self.layers = nn.Sequential(*[DenseLayer1D(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])

        # local feature fusion
        self.lff = nn.Conv1d(in_channels + growth_rate * num_layers, growth_rate, kernel_size=1)

    def forward(self, x):
        return x + self.lff(self.layers(x))  # local residual learning


class RDN1D(nn.Module):
    def __init__(self, scale_factor, num_channels, num_features, growth_rate, num_blocks, num_layers):
        super(RDN1D, self).__init__()
        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers

        # shallow feature extraction
        self.sfe1 = nn.Conv1d(num_channels, num_features, kernel_size=3, padding=3 // 2)
        self.sfe2 = nn.Conv1d(num_features, num_features, kernel_size=3, padding=3 // 2)

        # residual dense blocks
        self.rdb1Ds = nn.ModuleList([RDB1D(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdb1Ds.append(RDB1D(self.G, self.G, self.C))

        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv1d(self.G * self.D, self.G0, kernel_size=1),
            nn.Conv1d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )

        # up-sampling
        assert 2 <= scale_factor <= 4
        if scale_factor == 2 or scale_factor == 4:
            self.upscale = []
            for _ in range(scale_factor // 2):
                self.upscale.extend([nn.Conv1d(self.G0, self.G0 * (2 ** 2), kernel_size=3, padding=3 // 2),
                                     nn.PixelShuffle(2)])
            self.upscale = nn.Sequential(*self.upscale)
        else:
            self.upscale = nn.Sequential(
                nn.Conv1d(self.G0, self.G0 * (scale_factor ** 2), kernel_size=3, padding=3 // 2),
                nn.PixelShuffle(scale_factor)
            )

        self.output = nn.Conv1d(self.G0, num_channels, kernel_size=3, padding=3 // 2)

    def forward(self, x):
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)

        x = sfe2
        local_features = []
        for i in range(self.D):
            x = self.rdb1Ds[i](x)
            local_features.append(x)

        x = self.gff(torch.cat(local_features, 1)) + sfe1  # global residual learning
        x = self.upscale(x)
        x = self.output(x)
        return x
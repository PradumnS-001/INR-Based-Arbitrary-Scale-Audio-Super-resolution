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
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        state_dict = model.state_dict()
        for name in self.shadow:
            self.shadow[name].copy_(self.shadow[name] * self.decay + state_dict[name] * (1.0 - self.decay))

    @torch.no_grad()
    def apply_shadow(self, model):
        model.load_state_dict(self.shadow, strict=True)